import unittest


class GuiPageImportTests(unittest.TestCase):
    def test_application_pages_import(self):
        from src.gui.pages.application_page import ApplicationPage
        from src.gui.pages.congestion_warning_page import CongestionWarningPage
        from src.gui.pages.event_simulation_page import EventSimulationPage

        self.assertIsNotNone(ApplicationPage)
        self.assertIsNotNone(CongestionWarningPage)
        self.assertIsNotNone(EventSimulationPage)


if __name__ == "__main__":
    unittest.main()
