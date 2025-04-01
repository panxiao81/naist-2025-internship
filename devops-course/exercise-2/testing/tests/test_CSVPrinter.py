import unittest
from csvprinter.CSVPrinter import CSVPrinter


def setUpModule():
    print("Running setUpModule")

def tearDownModule():
    print("Running setDownModule")

class TestCSVPrinter(unittest.TestCase):

    def test_TestMethod1(self):
        printer = CSVPrinter("sample.csv")
        l = printer.read()
        self.assertEqual(3, len(l))

    def test_TestMethod2(self):
        printer = CSVPrinter("sample.csv")
        l = printer.read()
        self.assertEqual("value1A", l[0][0])

    def test_TestMethod3(self):
        printer = CSVPrinter("a-file-does-not-exist.csv")
        with self.assertRaises(FileNotFoundError):
            printer.read()

    @classmethod
    def setUpClass(cls):
        print("Running setUpClass")
        return super().setUpClass()
    
    @classmethod
    def tearDownClass(cls):
        print("Running tearDownClass")
        return super().tearDownClass()
    
    def setUp(self):
        print("Running setUp")
        return super().setUp()
    
    def tearDown(self):
        print("Running tearDown")
        return super().tearDown()
