# ASC File Processing Script

This Python script processes `.asc` files found in a user-specified directory. It reads the files, extracts a specific portion of the data (from a given start line to an end line), converts the data into CSV format, and generates corresponding plots saved as PNG images.

## Features

-   **User Prompts:**

    -   Select an input directory containing `.asc` files.
    -   Automatically create two subdirectories (`CSV` and `PNG`) inside the input directory for outputs.
    -   A custom dialog to enter processing parameters with an option to run default values:
        -   **Start Line:** 13
        -   **End Line:** 10814
        -   **Sampling Rate:** 4
        -   **Y-Axis Minimum:** 2000
        -   **Y-Axis Maximum:** 100000

-   **File Processing:**

    -   Converts selected lines from `.asc` files into CSV files.
    -   Generates plots of the signal data, applying the user-defined Y-axis limits, and saves them as PNG images.

-   **Dynamic File Encoding:**
    -   Uses `chardet` (if available) to detect file encoding.

## Prerequisites

-   Python 3.x
-   Required packages listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Infinity-Diamond/CE-Processing
    cd CE-Processing
    ```
