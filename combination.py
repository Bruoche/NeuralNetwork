import numpy as np
from typing import TypeVar, Generic

T = TypeVar('T')

class Options(Generic[T]):
	def __init__(self, values: list[T]) -> None:
		if len(values) <= 0:
			raise ValueError("The array to combine must not be empty")
		self.values = values
		self.index = 0

	def get(self) -> T:
		return self.values[self.index]

	def next(self) -> bool:
		if self.index >= len(self.values)-1:
			self.index = 0
			return True
		else:
			self.index += 1
			return False


class Iterator:
	def __init__(self, combinations: list[Options]) -> None:
		self.combinations = combinations

	def __iter__(self):
		self.a = 0
		return self

	def __next__(self):
		iteration = self.a
		self.a += 1
		if iteration <= 0:
			return iteration			
		index = 0
		looped = True
		while looped:
			if index >= len(self.combinations):
				raise StopIteration
			looped = self.combinations[index].next()
			index += 1
		return iteration