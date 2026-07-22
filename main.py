"""python main.py"""

import multiprocessing

from src.gui_app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
