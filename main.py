import os
import logging
import time
import PLAXIS.Input as InputModel
import PLAXIS.Output as OutputModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    
    model = InputModel.PlaxisModelInput()
    model.Create_Model()

    output = OutputModel.PlaxisModelOutput()
    output.GetOutput()
    OutputData = output.__output_data

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()