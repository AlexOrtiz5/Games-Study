# Game Insights: Analyzing Popularity, Platforms, and User Feedback

  Project status(Completed)
# Project objective

  The aim of this project is to perform a thorough study of a gaming dataset that includes user reviews, platform accessibility, metacritic scores, game popularity, and more. Finding patterns, correlations, and insights in the data that can help the gaming business make decisions is the main objective.
  
  - **Game Popularity vs. Features**: 
    Examine the connection between estimated owners and peak concurrent users (peak_ccu). Find out if more players are drawn to games with higher estimated ownership during peak periods.
  - **Impact of Platforms on Game Adoption**: 
    Examine how platform accessibility affects the uptake of games. Examine the ownership trends of games that are accessible on Linux, Mac, and Windows platforms.
  - **Metacritic Score and User Feedback**: 
    Analyze how well user and Metacritic scores match up. Determine whether the opinions of expert critics and users consistently correlate positively.
  - **Impact of Game Features on Reviews**: 
    Examine the connection between a game's amount of accomplishments and the type of reviews it receives. Examine whether reviews of games are generally more positive when they have more achievements.
  - **Playtime Patterns**: 
    Examine how average and median playtime for games relate to one another. Examine playtime analytics for any recurring trends that can reveal information about user involvement.
  - **Price Influence**: 
    Examine how user scores are affected by game prices. Examine if higher-priced games typically have better user reviews. 
  - **Categorization Impact on Popularity**: 
    Analyze the relationship between an estimated game's owners and the number of categories/genres it falls under. Examine if more broadly classified games typically have greater ownership rates.

  By addressing these goals, the project hopes to provide publishers, industry stakeholders, and game developers with practical insights that will improve their decision-making, marketing plans, and approaches to game production. The goal of the analysis is to identify trends that may help games succeed and gain traction in the ever-changing gaming industry.

# Methods

  List with methods:
  - Filtering
    - Column Renaming: To maintain uniformity, column names were changed to lowercase and underscores were used in place of spaces.
    - Managing Missing Values: Determined which columns had missing values and used suitable techniques to impute them.
    - Data Type Separation: For particular processing, columns were divided into number and text categories.
    - Numerical Column Imputation: The mean was used to fill in any missing values in numerical columns by using SimpleImputer.
    - Imputation of Text/Categorical Columns: To handle missing text/categorical data, used SimpleImputer with constant value 'unknown'.
    - Supported and Full Audio Languages Processing: In "supported_languages" and "full_audio_languages," brackets have been removed, and string formats have been standardised.
    - String Processing: To clean and standardize the data, brackets were removed from the designated columns.
    - Eliminating Outliers: Finding and eliminating data points that substantially differ from the majority.
    - Platform Filtering: Highlighting games compatible with Linux, Mac, and Windows.

  - Grouping
    - Grouping by Platforms: To examine multi-platform availability, data is gathered based on gaming platforms.
    - Grouping by Review Type: For comparative analysis, classify reviews as either positive or negative.
    - Grouping by Categories/Genres: Assigning each game to a certain number of categories and genres.

  - Analysis:
    - Filtering and Selection: Relevant columns were chosen for in-depth examination.
    - Grouping and Aggregation: For aggregated insights, data was grouped according to platforms, review types, and categories/genres.
    - Statistical Analysis: For numerous comparisons, the T-test and Pearson Correlation were run.
    - Feature Engineering: Counted categories and genres and created indications such as "Multiple Platforms."
    - Assessing the linear relationship between two continuous variables, such as peak CCU vs. estimated owners or Metacritic score vs. user score, is done using Pearson correlation analysis.
    - T-test: To ascertain whether there is a statistically significant difference, the means of two groups are compared.

  - Features:
    - Release Date Conversion: To facilitate analysis, the'release_date' field was converted to datetime format.
    Transformation of Estimated Owners: For uniformity, a transformation was applied to the 'estimated_owners' column.
    - Creating a "Multiple Platforms" Indicator: This involves combining data from the Linux, Mac, and Windows columns to show which games are available on multiple operating systems.
    - Counting Categories/Genres: To determine how many distinct categories and genres there are in total for every game, add a new column.

  - Visualization
    - Scatter Plots: Showing the relationship between variables with a visual aid, like metacritic score versus user score or estimated owners versus peak CCU.
    - Boxplots: Showing the distribution of estimated owners according to various platforms and pointing out possible anomalies.
    - Heatmap: A visual representation of the correlation matrix between variables that helps spot trends and connections.
    - Bar Plots: Showing the average scores for games with both favorable and unfavorable ratings.
    - Joint graphs: To see how user and Metacritic scores relate to one another, create joint graphs.

    - Tableau:
      - Tableau visualization: Tableau was used to create interactive dashboards and visualize data trends.
      - Duplicating Python Visualizations: Converted scatter plots, box plots, and other Python visualizations into Tableau.

    - Modeling:
      - PyCaret Modeling: Made use of the PyCaret library to compare, tune, and assess models quickly and effectively.

# Technologies 

  List with used technologies:
  - Python: The primary programming language for data analysis, visualization, and modeling.
  - Pandas: A powerful data manipulation library used for handling and analyzing structured data.
  - SQLAlchemy: SQL toolkit and Object-Relational Mapping (ORM) library for database interaction.
  - MySQL: Relational database management system used for storing and retrieving structured data.
  - PyCaret: A machine learning library in Python used for automating the end-to-end model development process.
  - Matplotlib and Seaborn: Python libraries for creating static, animated, and interactive visualizations.
  - Scikit-Learn: Machine learning library in Python used for various tasks such as regression, classification, clustering, etc.
  - NLTK (Natural Language Toolkit): Library for working with human language data, used for text data processing.
  - Tableau: Data visualization tool used for creating interactive and shareable dashboards.
  - SimpleImputer: Scikit-Learn module used for imputing missing values in the dataset.

# Project Description

  The focus of this research is a large gaming dataset that includes a variety of data pertaining to multiple games. Critical attributes like "AppID," "Name," "Release date," "Estimated owners," "Peak CCU," "Price," "Metacritic score," "User score," and more are included in the dataset. These features provide an in-depth look at the features, player engagement data, and important feedback indicators of every game.

  The sources of the dataset could include user reviews, Metacritic scores, gaming platforms, and other sites that support the gaming industry. The collection includes both qualitative and quantitative features, such as game categories, genres, and tags, in addition to quantitative parameters like estimated ownership, peak concurrent users, cost, and review scores.

  **Key characteristics of the dataset include**:
    - **Quantitative Metrics**: Measures that reveal information about a game's popularity and player involvement levels include peak CCU and estimated owners.
    - **User Feedback**: The reactions of the community to each game are reflected in user ratings, reviews (both good and negative), and recommendations.
    - **Metacritic ratings**: By including Metacritic ratings, it's possible to compare user opinions with the evaluations of professional critics.
    - **Platform Accessibility**: Details on the platforms (Windows, Mac, and Linux) that are available offer insights into the game's potential user base and reach.
    - **Playtime Patterns**: User engagement and the length of gaming sessions can be analyzed using average and median playtime data.

  Owing to its extensive scope, the dataset is a useful tool for performing various analysis aimed at identifying patterns, correlations, and trends in the gaming sector. Through the usage of these traits, the project hopes to learn more about user preferences, game popularity, and the effects of different features on a game's overall success.

# Steps

  ## Data Profiling Challenges: 
  Text-based columns presented difficulties for the ydata_profiling data profiling phase, which resulted in the rejection of some features from the study. Understanding the many data types and formats that profiling tools offer is crucial.

  1. To begin analysis, get the dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?select=games.csv).
  2. Open the `analyze.ipynb` notebook after downloading.
  3. Examine the dataset and note the kinds and quantity of columns.
  4. Create an HTML report by using `ydata_profiling` to profile data.
  5. Remove nine text-based columns because the profiling didn't function with them out of a total of 39 columns.
  6. Store the profiling report in the file named `resources`.

  ## Data Cleaning and Transformation: 
  Managing missing values, modifying column types, and standardizing column names were all part of the data cleaning procedure. In order to prepare the data for analysis and modeling, several steps are essential.

  1. To clean the data, open the `clean_data.ipynb` notebook.
  2. Modify column types, handle object columns, and use techniques to fill in missing values.
  3. Change the column names to lowercase and eliminate spaces.
  4. To facilitate MySQL import, divide the data into distinct CSV files.

  ## SQL Integration: 
  The steam_games.sql script's usage of SQL points to a relational database strategy. This improves data management overall by enabling effective data storage, retrieval, and querying.

  1. Run the `data/sql` folder's `steam_games.sql` file.
  2. For notebooks after this one, this step is required. Use other data for modeling and assumptions if SQL is skipped.
  3. To import cleaned data into SQL for more effective queries, go to `import_sql.ipynb`.

  ## Testing of Hypotheses: 
  Seven hypotheses pertaining to various facets of the data are being tested for this project. The knowledge gathered from these tests may be quite helpful in identifying patterns and relationships in the dataset.

  1. To test your hypothesis, go to `hypothesis_testing.ipynb`.
  2. Using graphs and tests, demonstrate seven distinct hypotheses for each category of data examined.

  ## PyCaret-Based Data Modeling: 
  Using PyCaret to model data indicates a desire to automate the process of machine learning. This can facilitate faster decision-making by accelerating the training and evaluation of numerous models.

  1. For data modeling, visit `data_modeling.ipynb`.
  2. Examine the prediction models that PyCaret generated for every hypothesis.
  3. Examine the `data/model` file's prediction error graphs, training data kinds, and model saving.

  ## Tableau Visualization: 
  Making powerful and engaging visual representations of the data is the emphasis of using Tableau for visualization. These visuals can help explain difficult concepts to a larger audience.

  1. Look in the `presentation/tableau` folder for Tableau graphics.
  2. Alternatively, navigate to the Visualization section below to see the Tableau dashboard.

  ## Data Exploration App: 
  The creation of an app that allows users to explore data in one location suggests that accessibility and user-friendly interaction are priorities. Stakeholders who might not be familiar with data analysis tools may find this useful.

  1. Use the app to view all data in one location and look through the `src` folder to see data presentation and test models.
  2. Alternatively, navigate to the Visualization section below to examine the streamlit app.

# Conclusion

  To sum up, this project explores a wealth of gaming data and provides a comprehensive viewpoint on a variety of gaming-related topics. We investigated connections, patterns, and trends across a range of dimensions through thorough studies, illuminating important variables affecting player engagement, user happiness, and game popularity.

  **Key Findings**: 
  - **Game Popularity and Features**: The study showed that there was a positive relationship between estimated owners and peak concurrent users (peak CCU), meaning that games with higher ownership levels typically draw in more concurrent players.
  - **Impact of Platforms on Adoption**: Games that are available on several different platforms (Windows, Mac, and Linux) show more ownership, indicating that a variety of platforms encourages more players to adopt games.
  - **Metacritic Score and User Feedback**: Against expectations, there was not a consistent strong positive association found in the correlation analysis between Metacritic scores and user scores. This highlights how subjective user judgments are in contrast to those of expert critics.
  - **Impact of Game Features on Reviews**: Positive reviews are associated with more achievements in games, suggesting a possible link between positive feedback and in-game achievements.
  - **Playtime Patterns**: Games with longer median playtimes also typically have longer average playtimes, according to a positive link between average and median playtimes.
  - **Price Influence**: Oddly, the investigation failed to find a discernible relationship between game prices and user scores. This implies that factors other than the game's price have an impact on user happiness.
  - **Categorization Impact on Popularity**: A game's estimated owners positively correlated with the number of categories and genres it belonged to, indicating that more broadly classified games are played by a wider audience.

  For everyone involved in the gaming industry, from publishers and developers to analysts and enthusiasts, this initiative offers insightful information. Comprehending the factors that drive game success can help make better strategic choices, improve user experiences, and support the ongoing development of the gaming industry.
  
# Presentation and Visualization

  - **PowerPoint**: Locanted on the presentation folder
  - **Tableau**: [Games Study Dashboard](https://public.tableau.com/views/Games_tableau/Games?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link)  
  - **Streamlit**: Need to run the code that is found in the `src` file named `app.py`.
  