from __future__ import annotations

import unittest

import torch

from pyrkm.classifier import CustomDataset, SimpleClassifier, train_classifier


class TestCustomDataset(unittest.TestCase):

    def setUp(self):
        self.data = torch.randn(100, 28, 28)
        self.targets = torch.randint(0, 10, (100, ))
        self.dataset = CustomDataset(self.data, self.targets)

    def test_length(self):
        self.assertEqual(len(self.dataset), 100)

    def test_getitem(self):
        img, label = self.dataset[0]
        self.assertEqual(img.shape, (28, 28))
        self.assertTrue(0 <= label < 10)


class TestSimpleClassifier(unittest.TestCase):

    def setUp(self):
        self.model = SimpleClassifier()

    def test_forward(self):
        x = torch.randn(1, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))


class TestTrainClassifier(unittest.TestCase):

    def setUp(self):
        self.train_data = torch.randn(100, 28, 28)
        self.train_targets = torch.randint(0, 10, (100, ))
        self.test_data = torch.randn(20, 28, 28)
        self.test_targets = torch.randint(0, 10, (20, ))
        self.train_set = (self.train_data, self.train_targets)
        self.test_set = (self.test_data, self.test_targets)

    def test_train_classifier(self):
        model, accuracy = train_classifier(self.test_set,
                                           self.train_set,
                                           num_epochs=1)
        self.assertIsInstance(model, SimpleClassifier)
        self.assertTrue(0 <= accuracy <= 100)


if __name__ == '__main__':
    unittest.main()
