# Games-Study

  Project status(Active, On-Hold, Completed)
# Project objective

  Paragraph with introduction/ objective of project

  The objective of this project is to conduct a comprehensive analysis of a gaming dataset encompassing various features, including game popularity, user feedback, platform accessibility, metacritic scores, and more. The primary goals are to uncover patterns, correlations, and insights within the data that can inform decision-making in the gaming industry.
  
  - **Game Popularity vs. Features**:
    Explore the relationship between peak concurrent users (peak_ccu) and estimated owners. Determine if games with higher estimated ownership tend to attract more players during peak times.
  - **Impact of Platforms on Game Adoption**:
    Investigate the influence of platform accessibility on game adoption. Analyze the ownership patterns of games available on multiple platforms (Windows, Mac, and Linux).
  - **Metacritic Score and User Feedback**:
    Examine the correlation between user scores and Metacritic scores. Understand if there is a positive correlation, indicating consistency between user perceptions and professional critics.
  - **Effect of Game Features on Reviews**:
    Assess the relationship between the number of achievements in games and the nature of their reviews. Explore whether games with more achievements tend to receive more favorable evaluations.
  - **Playtime Patterns**:
    Investigate the relationship between average playtime and median playtime for games. Determine if there are consistent patterns in playtime metrics, providing insights into user engagement.
  - **Price Influence**:
    Explore the impact of game pricing on user scores. Analyze whether games with higher prices tend to receive higher user ratings. 
  - **Categorization Impact on Popularity**:
    Examine the correlation between the number of categories/genres a game belongs to and its estimated owners. Investigate whether games with broader categorization tend to have higher ownership.

  By addressing these objectives, the project aims to offer actionable insights for game developers, publishers, and industry stakeholders to enhance decision-making, marketing strategies, and game development approaches. The analysis seeks to uncover patterns that can contribute to the success and popularity of games in the dynamic gaming landscape.

# Methods

  List with methods:
  - Filtering
    - Column Renaming: Converted column names to lowercase and replaced spaces with underscores for consistency.
    - Handling Missing Values: Identified columns with missing values and imputed them using appropriate strategies.
    - Data Type Separation: Segregated columns into numerical and text categories for specific processing.
    - Imputation of Numerical Columns: Utilized SimpleImputer to fill missing values in numerical columns with the mean.
    - Imputation of Text/Categorical Columns: Used SimpleImputer with constant value 'unknown' to handle missing text/categorical data.
    - Supported and Full Audio Languages Processing: Removed brackets and standardized string formats in 'supported_languages' and 'full_audio_languages.'
    - String Processing: Removed brackets from specified columns to clean and standardize data.
    - Selection of Relevant Columns: Choosing specific columns from the dataset for focused analysis.
    - Removing Outliers: Identifying and excluding data points that deviate significantly from the majority.
    - Filtering by Platforms: Focusing on games available on Windows, Mac, and Linux.

  - Grouping
    - Grouping by Platforms: Aggregating data based on the gaming platforms to analyze multi-platform availability.
    - Grouping by Review Type: Categorizing reviews as positive or negative for comparative analysis.
    - Grouping by Categories/Genres: Creating groups based on the number of categories and genres assigned to each game.

  - Analysis:
    - Filtering and Selection: Selected relevant columns for focused analysis.
    - Grouping and Aggregation: Grouped data by platforms, review type, and categories/genres for aggregated insights.
    - Statistical Analysis: Performed Pearson Correlation and T-test for various comparisons.
    - Feature Engineering: Created indicators like 'Multiple Platforms' and counted categories/genres.
    - Pearson Correlation: Assessing the linear relationship between two continuous variables, e.g., peak CCU vs. estimated owners, Metacritic score vs. user score.
    - T-test: Comparing means of two groups to determine if there is a statistically significant difference.

  - Features:
    - Release Date Conversion: Converted the 'release_date' column to datetime format for easier analysis.
    - Estimated Owners Transformation: Applied a transformation to the 'estimated_owners' column for uniformity.
    - Creating a 'Multiple Platforms' Indicator: Combining information from Windows, Mac, and Linux columns to identify games available on multiple platforms.
    - Counting Categories/Genres: Creating a new column to count the total number of unique categories and genres for each game.

  - Visualization
    - Scatter Plots: Visualizing the relationship between variables, such as estimated owners vs. peak CCU, and metacritic score vs. user score.
    - Boxplots: Illustrating the distribution of estimated owners based on different platforms and identifying potential outliers.
    - Heatmap: Displaying the correlation matrix between variables to identify patterns and relationships.
    - Bar Plots: Representing the mean achievements in games with positive and negative reviews.
    - Joint Plots: Creating joint plots to visualize the relationship between Metacritic scores and user scores.

    - Tableau:
      - Visualization in Tableau: Utilized Tableau for visualizing data patterns and creating interactive dashboards.
      - Replicating Python Visualizations: Translated Python visualizations (scatter plots, box plots, etc.) into Tableau.

    - Modeling:
      - PyCaret Modeling: Leveraged PyCaret library for streamlined and efficient model comparison, tuning, and evaluation.

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

  Paragraph with a description of the dataset, sources, characteristics ,etc.

  This project revolves around an extensive gaming dataset, encompassing diverse information related to numerous games. The dataset includes critical attributes such as 'AppID,' 'Name,' 'Release date,' 'Estimated owners,' 'Peak CCU,' 'Price,' 'Metacritic score,' 'User score,' and more. These attributes offer a comprehensive view of each game's characteristics, player engagement metrics, and critical feedback indicators.

  The dataset's sources may include gaming platforms, user reviews, Metacritic scores, and various other sources contributing to the gaming ecosystem. The characteristics of the dataset span quantitative metrics like estimated ownership, peak concurrent users, pricing, and review scores, as well as qualitative attributes such as game categories, genres, and tags.

  **Key characteristics of the dataset include**:
  - *Quantitative Metrics*: Metrics like estimated owners and peak CCU provide insights into a game's popularity and player engagement levels.
  - *User Feedback*: User scores, positive and negative reviews, and recommendations reflect the community's response to each game.
  - *Metacritic Scores*: The inclusion of Metacritic scores allows for a comparison between professional critics' assessments and user opinions.
  - *Platform Accessibility*: Information on platform availability (Windows, Mac, Linux) provides insights into the game's reach and potential user base.
  - *Playtime Patterns*: Average playtime and median playtime metrics reveal patterns in user engagement and the longevity of gaming sessions.

  The dataset's comprehensive nature makes it a valuable resource for conducting multifaceted analyses to uncover trends, correlations, and patterns within the gaming industry. The project aims to leverage these characteristics to gain insights into factors influencing game popularity, user preferences, and the impact of various features on a game's success.

# Steps
  Add here any insights you had during the project

# Conclusion
  Final conclusion

  In conclusion, this project delves into a rich gaming dataset, offering a holistic perspective on diverse aspects within the gaming industry. Through comprehensive analyses, we explored correlations, patterns, and trends across various dimensions, shedding light on critical factors influencing game popularity, player engagement, and user satisfaction.

  **Key Findings**:
  - *Game Popularity and Features*: The analysis revealed a positive correlation between peak concurrent users (peak CCU) and estimated owners, indicating that games with higher ownership tend to attract more concurrent players.
  - *Impact of Platforms on Adoption*: Games accessible across multiple platforms (Windows, Mac, Linux) exhibit broader ownership, suggesting that platform diversity contributes to increased game adoption.
  - *Metacritic Score and User Feedback*: Contrary to expectations, the correlation analysis between Metacritic scores and user scores did not consistently show a strong positive correlation. This emphasizes the nuanced nature of user opinions compared to professional critics.
  - *Effect of Game Features on Reviews*: Games with favorable reviews tend to have a higher quantity of achievements, pointing to a potential connection between positive reception and in-game accomplishments.
  - *Playtime Patterns*: The correlation between average playtime and median playtime revealed a positive relationship, indicating that games with longer median playtimes also tend to have longer average playtimes.
  - *Price Influence*: Surprisingly, the analysis did not establish a clear correlation between user scores and game prices. This suggests that user satisfaction is influenced by factors beyond the game's cost.
  - *Categorization Impact on Popularity*: The number of categories and genres a game belongs to showed a positive correlation with estimated owners, suggesting that games with broader categorization appeal to a larger audience.

  This project provides valuable insights for stakeholders in the gaming industry, from developers and publishers to analysts and enthusiasts. Understanding the dynamics influencing game success can inform strategic decisions, enhance user experiences, and contribute to the continuous evolution of the gaming landscape.
  
# Contact
  linkedin, github, medium, etc 