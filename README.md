# SCC 452 Group Project: MAS for Stocks
This project involved building a multi-layer MAS (multi-agent) system for stock advice. The system consists of 4 agents - `technical_agent`, `health_agent`, `sentiment_agent`, and `plotting_agent`. The system is tested on 3 different stocks - `Boeing`,`Apple`, and `Tesla`. 

The system's main objective is to outperform baseline approach and a standalone selected agent. The evaluation uses **Sharpe Ratio**, **Maximum Drawdown**, and **$ Return (in %)** together with visuals showing `portfolio's value` over time, `distribution of agents' votes`, and `agents' confidence` over time.


## Project Structure
* `final_code` contains only scripts needed for simulation, including simulation, and therefore simulation can be run directly in `simulation.ipynb` file. For simulation to run, navigate to the `simulation.ipynb` notebook and **run each cell separately**. Use **VPN** - after each of the 3 simulations for each company, switch to a different connection using VPN to bypass the API limit in `data_preparation` script. Data for sentiment agent is included there as well. All scripts are **in the same directory** and therefore simulation as well as **each agent** can be run directly.
* `individual_components` contain folders with each agent's script, a testing script, and additionally a documentation. These are raw files - all scripts **need to be in the same directory** for all scripts to run and work smoothly.
* `report.pdf` is the final report.

## Extensions Needed 
For the simulation to work, following libraries are needed:
```
pandas (version 2.3.3)
numpy (version 2.4.1)
scikit-learn (version 1.8.0)
yfinance (version 1.2.0)
matplotlib (version 3.10.8)
seaborn (version 0.13.2)
transformers (version 4.57.6)
```

Installation commands:
```
pip install pandas==2.3.3
pip install numpy==2.4.1
pip install scikit-learn==1.8.0
pip install yfinance==1.2.0
pip install matplotlib==3.10.8
pip install seaborn==0.13.2
pip install transformers==4.57.6
```

