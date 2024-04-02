#### Overview
This Jupyter notebook is designed to parse high-volatility news data from Investing.com, focusing on the USA. It automates the extraction of news items marked with high expected volatility and saves them into a SQLite database for further analysis or monitoring.

#### How to Use
1. **Prerequisites**: Ensure Python is installed along with the `requests`, `bs4`, and `sqlite3` libraries.
2. **Setting Up**: Clone/download the notebook and open it in a Jupyter environment.
3. **Running the Notebook**: Execute each cell sequentially. The initial cells will install necessary libraries (if not already installed) and establish a connection to Investing.com.
4. **Data Parsing**: The notebook will parse the economic calendar page for high-volatility news related to the USA and display the results.
5. **Saving Data**: Follow the instructions within the notebook to save the parsed data into a SQLite database.

#### Technical Details
- **Data Source**: [Investing.com Economic Calendar](https://www.investing.com/economic-calendar/)
- **Key Libraries**: `requests` for web requests, `BeautifulSoup` for HTML parsing, and `sqlite3` for database operations.
- **Customization**: Users can modify the notebook to adjust the filtering criteria or extend the database schema as needed.
