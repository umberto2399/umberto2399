# NBA Player Stats and Trade Analysis App

This project is a Streamlit-based web application designed to provide an interactive interface for viewing and analyzing NBA player statistics. Users can explore player data, add custom playing style descriptions, and analyze potential trade scenarios using OpenAI's GPT-4o API. The app aims to assist basketball managers and enthusiasts in making informed decisions about player trades and team compositions.

## Features

- **View Player Stats:** Display the entire dataset of player statistics, including filtering options by player or team.
- **Add Playing Style Descriptions:** Users can input and save custom playing style descriptions for individual players.
- **Trade Analysis:** Select players to give away and acquire, and generate trade insights using the ChatGPT API, which provides pros and cons for the proposed trades.

## Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NBA-trade-advisor.git
   cd NBA-trade-advisor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r req.txt
   ```

3. **Set up OpenAI API key:**
   - Replace `YOUR API KEY HERE` in the `client` initialization with your OpenAI API key.

4. **Run the app:**
   ```bash
   streamlit run GitHub_app.py
   ```

5. **Interact with the app:**
   - View and filter player stats.
   - Add and save playing style descriptions.
   - Analyze potential trades by selecting players to give away and acquire, and receive insights from the ChatGPT API.

## File Structure

- `app.py`: Main Streamlit application file.
- `player_stats.csv`: CSV file containing player statistics.
- `requirements.txt`: List of required Python packages.

## Dependencies

- pandas
- streamlit
- openai
