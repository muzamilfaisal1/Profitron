📈 AlgoBot – AI-Powered Crypto Trading Bot
AlgoBot is a cryptocurrency trading bot that enables users to build, test, optimize, simulate, and deploy trading strategies using a simple yet powerful Python-based framework. It features Telegram integration for remote monitoring and control, making crypto trading more accessible and efficient.

🚀 Features
View real-time market data and generate custom graphs

Configure and run backtests to evaluate strategies

Simulate trades with realistic market conditions

Deploy live bots with fully customizable parameters

Telegram integration for trade execution and stats on the go

Create take-profits, trailing stops, and custom stop-losses

Built-in optimizer to fine-tune strategy performance

Support for custom strategy development

🛠 Requirements
Python 3.7 to 3.9
TA-Lib (Technical Analysis Library)
Visual Studio Build Tools (for Windows users, if installation fails)

💻 Installation
Clone or extract the source code and run the following commands in the project directory:

*pip install pipenv
pipenv install*

🧱 TA-LIB Installation (Required)
Windows:
Download the appropriate .whl file for your Python version from here, then install it:

*pipenv shell
pip install <your_downloaded_whl_file>*

Linux/macOS:
Follow the official TA-LIB installation guide for your platform.

▶️ Running the Bot
To start AlgoBot:

*pipenv run bot*

To enable debug mode, set the environment variable:

*DEBUG=1*

📌 Note

Make sure you are using Python 3.7 – 3.9 only. Compatibility outside this range is not guaranteed.
