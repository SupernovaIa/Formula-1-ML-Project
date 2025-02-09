# Formula 1 Data Analysis and AI Chatbot

## 📜 Project overview

Formula 1 is one of the most competitive and technologically advanced sports in the world. Teams invest millions in collecting, analyzing, and applying detailed data on car performance, driver strategies, and race telemetry. 

However, much of this technical information is not effectively communicated to the general public, sports analysts, or potential sponsors, limiting their ability to understand performance insights and justify investments.

### Objective

The objective of this project is to analyze and effectively communicate the performance of drivers and teams throughout a Formula 1 season, leveraging insights derived from data and machine learning models.

This will help engage fans and enrich sports narratives by providing a clearer understanding of race performances and strategic decisions.

### Exploratory data analysis
- Analysis of general season results, including championship points, podiums, victories, and fastest laps.
- Detailed breakdown per Grand Prix, focusing on:
  - Qualifying and race performance.
  - Tire management and race strategies.
  - Lap-by-lap analysis of position changes.

### Machine learning models

- **Clustering of circuits** based on technical characteristics to identify similar track profiles.
- **Classification model for victory prediction**, using past race data and starting grid positions.

### AI Chatbot

- Implemented using **LangChain** and **vector databases**.
- Retrieves information from transcribed F1-related videos to answer user queries.
- Currently supports data from the **2024 Australian Grand Prix** as a proof of concept.

Formula 1 is one of the most competitive and technologically advanced sports in the world. Teams invest millions in collecting, analyzing, and applying detailed data on car performance, driver strategies, and race telemetry. 

However, much of this technical information is not effectively communicated to the general public, sports analysts, or potential sponsors, limiting their ability to understand performance insights and justify investments.

## 🏛️ Project Structure

```
Formula-1-ML-Project
├── data/       # Folder containing datasets
├── docs/       # Documentation files for RAG
├── model/      # Machine learning models
├── notebook/   # Jupyter Notebooks for EDA and modeling
├── pages/      # Streamlit application pages
├── src/        # Source code for data processing and model execution
├── .env        # Environment variables
├── .gitignore  # Git ignore file
├── App.py      # Main entry point for the Streamlit application
├── README.md   # Project documentation
```

## ⚙ Installation and requirements

This project was developed in Python 3.12. To set it up, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SupernovaIa/Formula-1-ML-Project
   ```
2. Navigate to the project directory:
   ```bash
   cd F1-Data-Analysis-Project
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebooks for data extraction, data analysis and machine learning models.
5. Launch the Streamlit dashboard:
   ```bash
   streamlit run App.py
   ```

### Required libraries:

- **OpenAI API Key:** Required for chatbot functionality. ([Documentation](https://platform.openai.com/))
- **ChromaDB:** Vector database for storing chatbot-related data ([Documentation](https://github.com/chroma-core/chroma)).
- **Category Encoders:** Encoding categorical variables for ML models ([Documentation](https://contrib.scikit-learn.org/category_encoders/)).

- **Pandas:** Data manipulation ([Documentation](https://pandas.pydata.org/))
- **NumPy:** Numerical data processing ([Documentation](https://numpy.org/))
- **Plotly:** Interactive visualizations ([Documentation](https://plotly.com/))
- **FastF1:** F1 telemetry and race data extraction ([Documentation](https://theoehrly.github.io/Fast-F1/))
- **Scikit-learn:** Machine Learning algorithms ([Documentation](https://scikit-learn.org/))
- **XGBoost:** Advanced ML models ([Documentation](https://xgboost.readthedocs.io/))
- **LangChain:** AI chatbot integration ([Documentation](https://python.langchain.com/))
- **Streamlit:** Web application framework ([Documentation](https://streamlit.io/))

## 🚀 Next steps

- Develop a **SQL database** to reduce dependency on FastF1 for data extraction.
- Improve clustering and classification models with additional data and feature selection.
- Utilize circuit clustering to encode `circuitId` in predictive models.
- Expand predictive modeling to forecast more race-related variables.
- Enhance the **dashboard** with new visualizations and interactivity.
- Add more **documents to the chatbot** to allow queries about different Grand Prix events.

## ✍️ Author

Javier Carreira - Lead Developer  
GitHub: [SupernovaIa](https://github.com/SupernovaIa)