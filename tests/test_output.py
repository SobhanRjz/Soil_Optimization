import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PLAXIS.Input import PlaxisModelInput
from PLAXIS.Output import PlaxisModelOutput
import unittest
from unittest.mock import patch, MagicMock

class TestPlaxisModelOutput(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set up Input patcher
        self.input_patcher = patch('PLAXIS.Input.plx.new_server')
        self.mock_input_server = self.input_patcher.start()
        self.mock_input_s_i = MagicMock()
        self.mock_input_g_i = MagicMock()
        self.mock_input_server.return_value = (self.mock_input_s_i, self.mock_input_g_i)

        # Set up Output patcher
        self.output_patcher = patch('PLAXIS.Output.plx.new_server')
        self.mock_output_server = self.output_patcher.start()
        self.mock_output_s_i = MagicMock()
        self.mock_output_g_i = MagicMock()
        self.mock_output_server.return_value = (self.mock_output_s_i, self.mock_output_g_i)

    def tearDown(self):
        """Clean up after each test method."""
        self.input_patcher.stop()
        self.output_patcher.stop()

    def test_model_flow(self):
        """Test the complete model flow from input to output."""
        # Create and run input model
        model = PlaxisModelInput()
        model.Create_Model()

        # Mock phase for output
        mock_phase = MagicMock()
        mock_phase.Name.value = "Phase_1"
        self.mock_output_g_i.Phases = [mock_phase]
        self.mock_output_g_i.getresults.side_effect = [
            [0, 1, 2],  # x_coords
            [0, 1, 2],  # y_coords
            [0.1, 0.2, 0.3],  # ux_values
            [0.1, 0.2, 0.3]   # uy_values
        ]

        # Create and run output model
        output = PlaxisModelOutput()
        output.GetOutput()

        # Verify input model calls
        self.mock_input_g_i.calculate.assert_called_once()
        self.mock_input_g_i.save.assert_called_once()
        
        # Verify output model calls
        self.assertTrue(self.mock_output_g_i.getresults.called)
        self.assertEqual(self.mock_output_g_i.getresults.call_count, 4)

    def test_model_flow_failure(self):
        """Test handling of failures in the model flow."""
        # Mock calculation failure
        self.mock_input_g_i.calculate.side_effect = Exception("Calculation failed")
        
        model = PlaxisModelInput()
        with self.assertRaises(Exception):
            model.Create_Model()

        # Should not proceed to output if input fails
        output = PlaxisModelOutput()
        with self.assertRaises(Exception):
            output.GetOutput()

if __name__ == '__main__':
    unittest.main()