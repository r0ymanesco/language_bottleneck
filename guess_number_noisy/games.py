import ipdb
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

from egg.core.interaction import LoggingStrategy


class NoisySymbolGameGS(nn.Module):
    """
    Implements one-symbol Sender/Receiver game. The loss must be differentiable wrt the parameters of the agents.
    Typically, this assumes Gumbel Softmax relaxation of the communication channel.
    >>> class Receiver(nn.Module):
    ...     def forward(self, x, _input=None):
    ...         return x
    >>> receiver = Receiver()
    >>> sender = nn.Sequential(nn.Linear(10, 10), nn.LogSoftmax(dim=1))
    >>> def mse_loss(sender_input, _1, _2, receiver_output, _3):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {}
    >>> game = SymbolGameGS(sender=sender, receiver=Receiver(), loss=mse_loss)
    >>> loss, interaction = game(torch.ones((2, 10)), None) #  the second argument is labels, we don't need any
    >>> interaction.aux
    {}
    >>> (loss > 0).item()
    1
    """

    def __init__(
            self,
            sender: nn.Module,
            receiver: nn.Module,
            channel: nn.Module,
            loss: Callable,
            train_logging_strategy: Optional[LoggingStrategy] = None,
            test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        """
        :param sender: Sender agent. sender.forward() has to output log-probabilities over the vocabulary.
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters: message and receiver_input.
        `message` is shaped as (batch_size, vocab_size).
        :param loss: Callable that outputs differentiable loss, takes the following parameters:
          * sender_input: input to Sender (comes from dataset)
          * message: message sent from Sender
          * receiver_input: input to Receiver from dataset
          * receiver_output: output of Receiver
          * labels: labels that come from dataset
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.
        """
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.channel = channel
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        channel_output = self.channel(message)
        receiver_output = self.receiver(channel_output, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction


class SymbolGameGS(nn.Module):
    """
    Implements one-symbol Sender/Receiver game. The loss must be differentiable wrt the parameters of the agents.
    Typically, this assumes Gumbel Softmax relaxation of the communication channel.
    >>> class Receiver(nn.Module):
    ...     def forward(self, x, _input=None):
    ...         return x
    >>> receiver = Receiver()
    >>> sender = nn.Sequential(nn.Linear(10, 10), nn.LogSoftmax(dim=1))
    >>> def mse_loss(sender_input, _1, _2, receiver_output, _3):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {}
    >>> game = SymbolGameGS(sender=sender, receiver=Receiver(), loss=mse_loss)
    >>> loss, interaction = game(torch.ones((2, 10)), None) #  the second argument is labels, we don't need any
    >>> interaction.aux
    {}
    >>> (loss > 0).item()
    1
    """

    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        """
        :param sender: Sender agent. sender.forward() has to output log-probabilities over the vocabulary.
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters: message and receiver_input.
        `message` is shaped as (batch_size, vocab_size).
        :param loss: Callable that outputs differentiable loss, takes the following parameters:
          * sender_input: input to Sender (comes from dataset)
          * message: message sent from Sender
          * receiver_input: input to Receiver from dataset
          * receiver_output: output of Receiver
          * labels: labels that come from dataset
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.
        """
        super(SymbolGameGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
