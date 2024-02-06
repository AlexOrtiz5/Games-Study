import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, spearmanr
import os
from pycaret.regression import *

# Specify the directory containing the CSV files
directory_path = '../data/clean/'

# Specify the order of CSV files
csv_order = ['game.csv', 'languages.csv', 'platform.csv', 'metacritic.csv', 'user_feedback.csv',
             'playtime.csv', 'development.csv', 'categorization.csv', 'media.csv']

# Initialize an empty DataFrame to store the merged result
game_data = pd.DataFrame()

# Loop through each CSV file, read it into a DataFrame, and merge it with the existing DataFrame
for csv_file in csv_order:
    file_path = os.path.join(directory_path, csv_file)
    df = pd.read_csv(file_path)

    # Merge based on the 'appid' column
    if game_data.empty:
        game_data = df
    else:
        game_data = pd.merge(game_data, df, on='appid', how='outer')

st.header("Game Insights: Analyzing Popularity, Platforms, and User Feedback")  

st.subheader("Data used for the analisis")      
st.write(game_data)

##################################################################################################################################
# Game Popularity vs. Features
if st.sidebar.checkbox('Hypothesis 1: Game Popularity vs. Features', True):
    st.header("1. Game Popularity vs. Features:")
    st.markdown("- **Theory**: Peak concurrent users (peak_ccu) of games are generally higher when their estimated owners are higher.")
    st.markdown('''- *Null Hypothesis*: Estimated owners and peak concurrent users do not significantly correlate.
    - *Alternative Hypothesis*: Peak concurrent users and estimated owners have a substantial positive association.''')

    # Convert columns to numeric and handle non-numeric values
    numeric_columns = ['estimated_owners', 'peak_ccu']
    games_data = game_data[['appid','estimated_owners', 'peak_ccu']].copy()
    games_data[numeric_columns] = games_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Plotting
    st.subheader("Scatter Plot and Heatmap Side by Side")

    # Scatter plot
    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='estimated_owners', y='peak_ccu', data=games_data, ax=scatter_ax)
    scatter_ax.set_xlabel('Estimated Owners')
    scatter_ax.set_ylabel('Peak CCU')

    # Heatmap
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(games_data[['estimated_owners', 'peak_ccu']].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=heatmap_ax)

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(scatter_fig)

    with col2:
        st.pyplot(heatmap_fig)

    # Drop rows with any NaN values (non-numeric after conversion)
    games_data = games_data.dropna(subset=numeric_columns)

    # Detect and handle outliers
    Q1 = games_data.drop(columns=['appid', 'estimated_owners']).quantile(0.25)
    Q3 = games_data.drop(columns=['appid', 'estimated_owners']).quantile(0.75)
    IQR = Q3 - Q1

    games_data_no_outliers = games_data[~((games_data.drop(columns=['appid', 'estimated_owners']) < (Q1 - 3.5 * IQR)) | 
                                        (games_data.drop(columns=['appid', 'estimated_owners']) > (Q3 + 3.5 * IQR))).any(axis=1)]

    # Plotting
    st.subheader("Scatter Plot and Heatmap Side by Side (No Outlier)")

    # Scatter plot without outliers
    no_outliers_fig, no_outliers_ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='estimated_owners', y='peak_ccu', data=games_data_no_outliers, ax=no_outliers_ax)
    no_outliers_ax.set_xlabel('Estimated Owners')
    no_outliers_ax.set_ylabel('Peak CCU')

    # Heatmap without outliers
    no_heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(games_data_no_outliers[['estimated_owners', 'peak_ccu']].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=heatmap_ax)

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(no_outliers_fig)

    with col2:
        st.pyplot(no_heatmap_fig)
    # Calculate Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(games_data_no_outliers['estimated_owners'], games_data_no_outliers['peak_ccu'])
    st.subheader(f'Pearson Correlation Coefficient and P-Value')
    st.write(f'The Pearson Correlation Coefficient is: {correlation}')
    st.write(f'The Pearson P-Value is: {p_value}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value < 0.05:
        st.write("Reject the null hypothesis as a result. A notable association exists.")
    else:
        st.write("Consequently, the null hypothesis is not rejected. Not a very strong link.")
        
    # Load the pickled model
    model_path = '../data/model/hypothesis1'
    games_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(games_model, data=games_data))

    # Create a simple form for user input
    st.header("Game Popularity Prediction")
    st.subheader("Enter Game Features:")
    appid = st.number_input("AppID", min_value=0, value=100000)
    peak_ccu = st.number_input("Peak CCU", min_value=0, value=100000)
    estimated_owners = st.number_input("Estimated Owners", min_value=0, value=100000)

    # Ensure order and names match the model's expectations
    user_input = pd.DataFrame({
        'appid': [appid],
        'peak_ccu': [peak_ccu],
        'estimated_owners': [estimated_owners]
        # Add more columns based on the features used in your model
    })

    # Display user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions using the loaded model
    predicted_peak_ccu = games_model.predict(user_input[['appid', 'peak_ccu']])

    # Function to interpret predicted peak concurrent users
    def interpret_prediction(predicted_value):
        if predicted_value > 0:
            return "The predicted peak concurrent users suggest an increase in popularity."
        elif predicted_value < 0:
            return "The predicted peak concurrent users suggest a decrease in popularity."
        else:
            return "The predicted peak concurrent users suggest no significant change in popularity."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted Peak Concurrent Users:")
    st.write(interpret_prediction(predicted_peak_ccu[0]))
    st.write('Number predicted: ', predicted_peak_ccu[0].round(2))

    # # Scatter plot
    # scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    # sns.scatterplot(x='estimated_owners', y='peak_ccu', data=games_data, ax=scatter_ax)
    # scatter_ax.set_xlabel('Estimated Owners')
    # scatter_ax.set_ylabel('Peak CCU')

    # # Add a point for the user input and predicted value
    # scatter_ax.scatter(user_input['estimated_owners'], predicted_peak_ccu[0], color='red', marker='x', label='User Input & Prediction')
    # scatter_ax.legend()

    # # Display the plot
    # st.subheader("Scatter Plot with User Input and Prediction")
    # st.pyplot(scatter_fig)
##################################################################################################################################
# Impact of Platforms on Game Adoption
if st.sidebar.checkbox('Hypothesis 2: Impact of Platforms on Game Adoption', False):
    st.header("2. Impact of Platforms on Game Adoption:")
    st.markdown("- **Theory**: There are more players in games that are accessible on several platforms (Windows, Mac, and Linux).")
    st.markdown('''- *Null Hypothesis*: The number of players in games that are accessible on various platform combinations does not significantly differ from one another.
    - *Alternative Hypothesis*: Compared to games on a single platform, games available on many platforms have a larger user base.''')

    platform_df = game_data[['estimated_owners','peak_ccu', 'windows', 'mac', 'linux']].copy()
    platform_df['platform'] = platform_df[['windows', 'mac', 'linux']].sum(axis=1) > 1

    # Boxplot for Multiple Platforms and Windows side by side
    st.subheader("Boxplots: Estimated Owners by Multiple Platforms and Windows")

    # Boxplot for Multiple Platforms
    boxplot_fig, boxplot_ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='platform', y='estimated_owners', data=platform_df, ax=boxplot_ax)
    boxplot_ax.set_title('Estimated Owners by Platform')
    boxplot_ax.set_xlabel('Multiple Platforms')
    boxplot_ax.set_ylabel('Estimated Owners')

    # Boxplot for Windows
    windows_boxplot_fig, windows_boxplot_ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='windows', y='estimated_owners', data=platform_df, ax=windows_boxplot_ax)
    windows_boxplot_ax.set_title('Estimated Owners on Windows')
    windows_boxplot_ax.set_xlabel('Windows')
    windows_boxplot_ax.set_ylabel('Estimated Owners')

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(boxplot_fig)

    # Boxplot for Windows
    with col2:
        st.pyplot(windows_boxplot_fig)
        
    # Detect and handle outliers
    platform_df_no_outliers = platform_df[(platform_df['estimated_owners'] < 3500000.0)]

    # Countplot without outliers
    st.subheader("Countplots: Estimated Owners by Multiple Platforms (No Outliers)")
    no_countplot_fig, countplot_ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='estimated_owners', data=platform_df_no_outliers, hue=platform_df_no_outliers['platform'].astype('str'), ax=countplot_ax)
    countplot_ax.set_title('Estimated Owners by Platform (No Outliers)')
    countplot_ax.set_xlabel('Multiple Platforms')
    countplot_ax.set_ylabel('Estimated Owners')
    st.pyplot(no_countplot_fig)
    
    # T-test for Windows vs. Multiple Platforms
    t_statistic, p_value_ttest = ttest_ind(platform_df_no_outliers['estimated_owners'][platform_df_no_outliers['windows'].eq(1)],
                            platform_df_no_outliers['estimated_owners'][platform_df_no_outliers['platform'].eq(True)],
                            equal_var=False)  # Assuming unequal variances

    # Display T-test results
    st.subheader("T-test Results: Windows vs. Multiple Platforms")
    st.write(f'The T-statistic is: {t_statistic}')
    st.write(f'The P-Value for the T-test is: {p_value_ttest}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value_ttest < 0.05:
        st.write("Reject the null hypothesis as a result. The player base on Windows and Multiple Platforms differs significantly.")
    else:
        st.write("Consequently, the null hypothesis is not rejected. There isn't much of a player base difference between Windows and several platforms.")
        
    # Load the pickled model
    model_path = '../data/model/hypothesis2'
    platform_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(platform_model, data=platform_df))

    # User Input for Platform Prediction
    st.header("Platform Adoption Prediction")
    st.subheader("Enter Game Features:")
    estimated_owners = st.number_input("Estimated Owners", min_value=0, value=100000, key="owners_input")
    peak_ccu = st.number_input("Peak CCU", min_value=0, value=100000, key="ccu_input")
    windows = st.checkbox("Windows", value=True, key="windows_checkbox")
    mac = st.checkbox("Mac", value=False, key="mac_checkbox")
    linux = st.checkbox("Linux", value=False, key="linux_checkbox")
    platforms = st.checkbox("Platforms")

    # Ensure order and names match the model's expectations
    user_input_platform = pd.DataFrame({
        'estimated_owners': [estimated_owners],
        'peak_ccu': [peak_ccu],
        'windows': [windows],
        'mac': [mac],
        'linux': [linux],
        'platform': [platforms]
    })

    # Display user input
    st.subheader("User Input for Platform Prediction:")
    st.write(user_input_platform)

    # Make predictions using the loaded model
    predicted_platform = platform_model.predict(user_input_platform[['peak_ccu', 'windows', 'mac', 'linux', 'platform']])

    # Function to interpret predicted platform adoption
    def interpret_platform_prediction(predicted_value):
        if predicted_value > 0.5:
            return "The predicted platform adoption suggests a higher likelihood of the game being adopted on multiple platforms."
        elif predicted_value <= 0.5:
            return "The predicted platform adoption suggests a higher likelihood of the game being adopted on a single platform."
        else:
            return "The predicted platform adoption is uncertain."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted Platform Adoption:")
    st.write(interpret_platform_prediction(predicted_platform[0]))
    st.write('Number predicted: ',predicted_platform[0].round(2))

    # # Bar Plot for Predicted Platform Adoption
    # predicted_platform_fig, predicted_platform_ax = plt.subplots(figsize=(8, 6))
    # predicted_platform_ax.bar(user_input_platform.columns[2:], predicted_platform[0], color=['blue', 'orange', 'green'])
    # predicted_platform_ax.set_title('Predicted Platform Adoption')
    # predicted_platform_ax.set_xlabel('Platform')
    # predicted_platform_ax.set_ylabel('Predicted Adoption')

    # # Display the plot
    # st.subheader("Bar Plot for Predicted Platform Adoption")
    # st.pyplot(predicted_platform_fig)
##################################################################################################################################
# Metacritic Score and User Feedback
if st.sidebar.checkbox('Hypothesis 3: Metacritic Score and User Feedback', False):
    st.header("3. Metacritic Score and User Feedback:")
    st.markdown("- *Theory*: The user and Metacritic scores are positively correlated.")
    st.markdown('''- *Null Hypothesis*: There is no discernible relationship (ρ = 0) between user and Metacritic scores.
    - *Alternative Hypothesis*: User scores and Metacritic scores have a substantial positive connection (ρ > 0).''')

    user_score_df = game_data[['metacritic_score', 'user_score']].copy()

    # Scatter plot and Joint plot
    st.header("Metacritic Scores vs. User Scores Analysis")

    # Scatter plot
    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='metacritic_score', y='user_score', data=user_score_df, ax=scatter_ax)
    scatter_ax.set_title('Scatter Plot of Metacritic Scores vs. User Scores')
    scatter_ax.set_xlabel('Metacritic Score')
    scatter_ax.set_ylabel('User Score')
    scatter_ax.grid(True)

    # Joint plot or Regression plot
    jointplot_fig = sns.jointplot(x='metacritic_score', y='user_score', data=user_score_df, kind='reg')
    jointplot_fig.fig.suptitle('Joint Plot of Metacritic Scores vs. User Scores', y=1.02)

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(scatter_fig)

    # Boxplot for Windows
    with col2:
        st.pyplot(jointplot_fig)

    # Detect and handle outliers    
    user_score_df_no_outliers = user_score_df[(user_score_df['user_score'] > 0.0)]

    # Scatter plot and Joint plot
    st.header("Metacritic Scores vs. User Scores Analysis (No Outliers)")

    # Scatter plot
    no_scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='metacritic_score', y='user_score', data=user_score_df_no_outliers, ax=scatter_ax)
    scatter_ax.set_title('Scatter Plot of Metacritic Scores vs. User Scores')
    scatter_ax.set_xlabel('Metacritic Score')
    scatter_ax.set_ylabel('User Score')
    scatter_ax.grid(True)

    # Joint plot or Regression plot
    no_jointplot_fig = sns.jointplot(x='metacritic_score', y='user_score', data=user_score_df_no_outliers, kind='reg')
    no_jointplot_fig.fig.suptitle('Joint Plot of Metacritic Scores vs. User Scores', y=1.02)

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(no_scatter_fig)

    # Boxplot for Windows
    with col2:
        st.pyplot(no_jointplot_fig)
        
    # Calculate Spearman correlation coefficient and p-value
    correlation, p_value = spearmanr(user_score_df_no_outliers['metacritic_score'], user_score_df_no_outliers['user_score'])
    st.subheader("Spearman Correlation Coefficient and Hypothesis Testing")
    st.write(f'The Spearman Correlation Coefficient is: {correlation}')
    st.write(f'The Spearman P-Value is: {p_value}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value < 0.05:
        st.write("Reject the null hypothesis as a result. A notable positive association is present.")
    else:
        st.write("Consequently, the null hypothesis is not rejected. Not a very strong link.")  
              
    # Load the pickled model
    model_path = '../data/model/hypothesis3'
    user_score_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(user_score_model, data=user_score_df))

    # User Input for Model Prediction
    st.header("User Input for Metacritic Score and User Feedback Prediction")
    st.subheader("Enter Game Features:")

    metacritic_score = st.number_input("Metacritic Score", min_value=0, max_value=100, value=80)
    user_score = st.number_input("User Score", min_value=0, max_value=10, value=8)

    # Ensure order and names match the model's expectations
    user_input = pd.DataFrame({
        'metacritic_score': [metacritic_score],
        'user_score': [user_score]
        # Add more columns based on the features used in your model
    })

    # Display user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions using the loaded model
    predicted_user_score = user_score_model.predict(user_input)

    # Function to interpret predicted user score
    def interpret_user_score_prediction(predicted_value):
        if predicted_value >= 7.0:
            return "The predicted user score suggests a high likelihood of positive user feedback."
        elif 5.0 <= predicted_value < 7.0:
            return "The predicted user score suggests moderate user feedback."
        else:
            return "The predicted user score suggests a lower likelihood of positive user feedback."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted User Score:")
    st.write(interpret_user_score_prediction(predicted_user_score[0]))
    st.write('Number predicted: ',predicted_user_score[0].round(2))

    # # Plotting
    # scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    # sns.scatterplot(x='metacritic_score', y='user_score', data=user_score_df, ax=scatter_ax)
    # scatter_ax.set_title('Scatter Plot of Metacritic Scores vs. User Scores')
    # scatter_ax.set_xlabel('Metacritic Score')
    # scatter_ax.set_ylabel('User Score')
    # scatter_ax.grid(True)

    # # Add a point for the user input and predicted value
    # scatter_ax.scatter(metacritic_score, user_score, color='red', marker='x', label='User Input')
    # scatter_ax.scatter(predicted_user_score[0], user_score, color='green', marker='o', label='Predicted Value')
    # scatter_ax.legend()

    # # Display the plot
    # st.subheader("Scatter Plot with User Input and Prediction")
    # st.pyplot(scatter_fig)
##################################################################################################################################
# Effect of Game Features on Reviews
if st.sidebar.checkbox('Hypothesis 4: Effect of Game Features on Reviews', False):
    st.header("4. Effect of Game Features on Reviews:")
    st.markdown("- *Theory*: There are more achievements in games that have received favorable reviews.")
    st.markdown('''- *Null Hypothesis*: The number of accomplishments and the number of good evaluations do not significantly correlate.
    - *Alternative Hypothesis*: There are more achievements in games that have received positive reviews.''')

    feedback_df = game_data[['positive', 'negative', 'achievements']].copy()

    # Scatter plot and Heatmap
    st.header("User Feedback Analysis and Mean Achievements by Review Type")

    # Heatmap
    correlation_matrix = feedback_df[['positive', 'negative', 'achievements']].corr()
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1, vmax=1, ax=heatmap_ax)
    heatmap_ax.set_title('Correlation Heatmap')

    # Bar Plot for Mean Achievements by Review Type
    review_types = ['Positive', 'Negative']
    mean_achievements = [feedback_df[feedback_df['positive'] > 0]['achievements'].mean(),
                        feedback_df[feedback_df['negative'] > 0]['achievements'].mean()]

    bar_fig, bar_ax = plt.subplots(figsize=(8, 6))
    bar_ax.bar(review_types, mean_achievements, color=['green', 'red'])
    bar_ax.set_title('Mean Achievements in Games with Positive and Negative Reviews')
    bar_ax.set_xlabel('Review Type')
    bar_ax.set_ylabel('Mean Achievements')

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(heatmap_fig)

    # Boxplot for Windows
    with col2:
        st.pyplot(bar_fig)
        
    # Detect and handle outliers
    Q1 = feedback_df['achievements'].quantile(0.25)
    Q3 = feedback_df['achievements'].quantile(0.75)
    IQR = Q3 - Q1

    feedback_df_no_outliers = feedback_df[(feedback_df['achievements'] >= Q1 - 1.5 * IQR) & (feedback_df['achievements'] <= Q3 + 1.5 * IQR)]

    # Scatter plot and Heatmap
    st.header("User Feedback Analysis and Mean Achievements by Review Type (No Outliers)")

    # Heatmap
    correlation_matrix = feedback_df_no_outliers[['positive', 'negative', 'achievements']].corr()
    no_heatmap_fig, heatmap_ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1, vmax=1, ax=heatmap_ax)
    heatmap_ax.set_title('Correlation Heatmap')

    # Bar Plot for Mean Achievements by Review Type
    review_types = ['Positive', 'Negative']
    mean_achievements = [feedback_df_no_outliers[feedback_df_no_outliers['positive'] > 0]['achievements'].mean(),
                        feedback_df_no_outliers[feedback_df_no_outliers['negative'] > 0]['achievements'].mean()]

    no_bar_fig, bar_ax = plt.subplots(figsize=(8, 6))
    bar_ax.bar(review_types, mean_achievements, color=['green', 'red'])
    bar_ax.set_title('Mean Achievements in Games with Positive and Negative Reviews')
    bar_ax.set_xlabel('Review Type')
    bar_ax.set_ylabel('Mean Achievements')

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(no_heatmap_fig)

    # Boxplot for Windows
    with col2:
        st.pyplot(no_bar_fig)
        
    # Calculate Pearson correlation coefficient and p-value
    st.subheader("Pearson Correlation Coefficient and Hypothesis Testing")
    correlation, p_value = pearsonr(feedback_df_no_outliers['positive'], feedback_df_no_outliers['achievements'])
    st.write(f'The Pearson Correlation Coefficient is: {correlation}')
    st.write(f'The Pearson P-Value is: {p_value}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value < 0.05:
        st.write("The null hypothesis is rejected. A noteworthy link exists.")
    else:
        st.write("The null hypothesis is not successfully rejected. Not a big relationship.")   
             
    # Load the pickled model
    model_path = '../data/model/hypothesis4'
    feedback_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(feedback_model, data=feedback_df))

    # User input for the model prediction
    st.header("Feedback and Achievements Prediction")
    st.subheader("Enter User Feedback and Achievements:")
    positive_reviews = st.number_input("Number of Positive Reviews", min_value=0, value=100)
    negative_reviews = st.number_input("Number of Negative Reviews", min_value=0, value=10)
    achievements = st.number_input("Number of Achievements", min_value=0, value=20)

    # Ensure order and names match the model's expectations
    user_input = pd.DataFrame({
        'positive': [positive_reviews],
        'negative': [negative_reviews],
        'achievements': [achievements]
        # Add more columns based on the features used in your model
    })

    # Display user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions using the loaded model
    predicted_feedback = feedback_model.predict(user_input[['negative', 'achievements']])

    # Function to interpret predicted feedback
    def interpret_feedback_prediction(predicted_value):
        if predicted_value >= 0.5:
            return "The predicted feedback suggests a higher likelihood of positive reviews."
        else:
            return "The predicted feedback suggests a higher likelihood of negative reviews."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted Feedback:")
    st.write(interpret_feedback_prediction(predicted_feedback[0]))
    st.write('Number predicted: ',predicted_feedback[0].round(2))

    # # Bar Plot for Mean Achievements by Review Type
    # review_types = ['Positive', 'Negative']
    # mean_achievements = [feedback_df[feedback_df['positive'] > 0]['achievements'].mean(),
    #                      feedback_df[feedback_df['negative'] > 0]['achievements'].mean()]

    # bar_fig, bar_ax = plt.subplots(figsize=(8, 6))
    # bar_ax.bar(review_types, mean_achievements, color=['green', 'red'])
    # bar_ax.set_title('Mean Achievements in Games with Positive and Negative Reviews')
    # bar_ax.set_xlabel('Review Type')
    # bar_ax.set_ylabel('Mean Achievements')

    # # Add a point for the user input and predicted value
    # bar_ax.bar(['User Input'], [achievements], color='blue', label='User Input')
    # bar_ax.bar(['Predicted Value'], [predicted_feedback[0]], color='purple', label='Predicted Value')
    # bar_ax.legend()

    # # Display the plot
    # st.subheader("Bar Plot with User Input and Prediction")
    # st.pyplot(bar_fig)
##################################################################################################################################
# Playtime Patterns
if st.sidebar.checkbox('Hypothesis 5: Playtime Patterns', False):
    st.header("5. Playtime Patterns:")
    st.markdown("- *Theory*: Games with longer median playtimes also have longer average playtimes.")
    st.markdown('''- *Null Hypothesis*: The average playtime and median playtime do not significantly correlate.
    - *Alternative Hypothesis*: Games with longer median playtimes also have longer average playtimes over time.''')

    playtime_df = game_data[['appid', 'average_playtime_forever', 'median_playtime_forever']].copy()

    # Box Plot and Statistical Analysis
    st.header("Playtime Metrics Analysis")

    # Box Plot
    st.subheader("Box Plot of Playtime Metrics")
    boxplot_fig, boxplot_ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=playtime_df[['average_playtime_forever', 'median_playtime_forever']], ax=boxplot_ax)
    boxplot_ax.set_title('Box Plot of Playtime Metrics')
    boxplot_ax.set_ylabel('Playtime (minutes)')
    st.pyplot(boxplot_fig)

    # Detect and handle outliers
    playtime_df_no_outliers = playtime_df[(playtime_df['average_playtime_forever'] > 0.0)]

    # Box Plot
    st.subheader("Box Plot of Playtime Metrics (No Outliers)")
    no_boxplot_fig, boxplot_ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=playtime_df_no_outliers[['average_playtime_forever', 'median_playtime_forever']], ax=boxplot_ax)
    boxplot_ax.set_title('Box Plot of Playtime Metrics')
    boxplot_ax.set_ylabel('Playtime (minutes)')
    st.pyplot(no_boxplot_fig)
    
    # Perform Pearson correlation test and P_value
    st.subheader("Pearson Correlation Coefficient and Hypothesis Testing")
    correlation, p_value = pearsonr(playtime_df_no_outliers['average_playtime_forever'], playtime_df_no_outliers['median_playtime_forever'])
    st.write(f'The Pearson Correlation Coefficient is: {correlation}')
    st.write(f'The Pearson P-Value is: {p_value}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value < 0.05:
        st.write("Dismiss the null hypothesis. A noteworthy link exists.")
    else:
        st.write("Reject the null hypothesis ineffectively. Not a big relationship.")
        
    # Load the pickled model
    model_path = '../data/model/hypothesis5'
    playtime_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(playtime_model, data=playtime_df))

    # User input for the model prediction
    st.header("Playtime Prediction")
    st.subheader("Enter Playtime Metrics:")
    appid = st.number_input("AppID (minutes)", min_value=0, value=100000)
    average_playtime = st.number_input("Average Playtime (minutes)", min_value=0, value=60)
    median_playtime = st.number_input("Median Playtime (minutes)", min_value=0, value=30)

    # Ensure order and names match the model's expectations
    user_input = pd.DataFrame({
        'appid': [appid], 
        'average_playtime_forever': [average_playtime],
        'median_playtime_forever': [median_playtime]
        # Add more columns based on the features used in your model
    })

    # Display user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions using the loaded model
    predicted_playtime = playtime_model.predict(user_input[['median_playtime_forever']])

    # Function to interpret predicted playtime
    def interpret_playtime_prediction(predicted_value):
        if predicted_value >= 60:
            return "The predicted playtime suggests a longer playtime, indicating potential engagement with the game."
        elif 30 <= predicted_value < 60:
            return "The predicted playtime suggests a moderate playtime, indicating moderate engagement with the game."
        else:
            return "The predicted playtime suggests a shorter playtime, indicating lower engagement with the game."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted Playtime:")
    st.write(interpret_playtime_prediction(predicted_playtime[0]))
    st.write('Number predicted: ',predicted_playtime[0].round(2))

    # # Box Plot
    # boxplot_fig, boxplot_ax = plt.subplots(figsize=(8, 6))
    # sns.boxplot(data=playtime_df[['average_playtime_forever', 'median_playtime_forever']], ax=boxplot_ax)
    # boxplot_ax.set_title('Box Plot of Playtime Metrics')

    # # Add a point for the user input and predicted value
    # boxplot_ax.scatter([0], [average_playtime], color='blue', label='User Input (Average Playtime)')
    # boxplot_ax.scatter([1], [median_playtime], color='red', label='User Input (Median Playtime)')
    # boxplot_ax.scatter([2], [predicted_playtime[0]], color='purple', label='Predicted Value')
    # boxplot_ax.legend()

    # # Display the plot
    # st.subheader("Box Plot with User Input and Prediction")
    # st.pyplot(boxplot_fig)
##################################################################################################################################
# Price Influence
if st.sidebar.checkbox('Hypothesis 6: Price Influence', False):
    st.header("6. Price Influence:")
    st.markdown("- *Theory*: User scores for games are typically higher when they are more expensive.")
    st.markdown('''- *Null Hypothesis*: User scores and game pricing do not significantly correlate.
    - *Alternative Hypothesis*: User scores significantly positively correlate with games that cost more money.''')

    price_df = game_data[['appid', 'name', 'price', 'user_score']].copy()

    # Scatter plot and Statistical Analysis
    st.header("Price vs. User Score Analysis")

    # Scatter plot
    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    scatter_ax.scatter(price_df['price'], price_df['user_score'])
    scatter_ax.set_xlim((0, 100))
    scatter_ax.set_title('Scatter Plot of Price vs. User Score')
    scatter_ax.set_xlabel('Price')
    scatter_ax.set_ylabel('User Score')
    scatter_ax.grid(True)
    st.pyplot(scatter_fig)

    # Detect and handle outliers
    Q1 = price_df['price'].quantile(0.25)
    Q3 = price_df['price'].quantile(0.75)
    IQR = Q3 - Q1

    price_df_no_outliers = price_df[(price_df['price'] >= Q1 - 1.5 * IQR) & (price_df['price'] <= Q3 + 1.5 * IQR)]

    # Scatter plot and Statistical Analysis
    st.header("Price vs. User Score Analysis (No Outliers)")

    # Scatter plot
    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    scatter_ax.scatter(price_df_no_outliers['price'], price_df_no_outliers['user_score'])
    scatter_ax.set_xlim((0, 100))
    scatter_ax.set_title('Scatter Plot of Price vs. User Score')
    scatter_ax.set_xlabel('Price')
    scatter_ax.set_ylabel('User Score')
    scatter_ax.grid(True)
    st.pyplot(scatter_fig)
    
    # Calculate the correlation coefficient and p-value
    st.subheader("Pearson Correlation Coefficient and Hypothesis Testing")
    correlation, p_value = pearsonr(price_df_no_outliers['price'], price_df_no_outliers['user_score'])
    st.write(f'The Pearson Correlation Coefficient is: {correlation}')
    st.write(f'The Pearson P-Value is: {p_value}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value < 0.05:
        st.write("The null hypothesis is rejected. A notable association exists.")
    else:
        st.write("The null hypothesis is not successfully rejected. Not a very strong link.")
        
    # Load the pickled model
    model_path = '../data/model/hypothesis6'
    price_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(price_model, data=price_df))

    # User input for the model prediction
    st.header("Price vs. User Score Prediction")
    st.subheader("Enter Game Information:")
    game_price = st.number_input("Price", min_value=0.0, value=30.0, step=1.0)
    user_score = st.number_input("User Score", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    # Ensure order and names match the model's expectations
    user_input = pd.DataFrame({
        'price': [game_price],
        'user_score': [user_score]
        # Add more columns based on the features used in your model
    })

    # Display user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions using the loaded model
    predicted_user_score = price_model.predict(user_input[['price']])

    # Function to interpret predicted user score
    def interpret_user_score_prediction(predicted_value):
        if predicted_value >= 8.0:
            return "The predicted user score suggests a high level of satisfaction. Players are likely to find the game very enjoyable."
        elif 6.0 <= predicted_value < 8.0:
            return "The predicted user score indicates a moderate level of satisfaction. Players may find the game enjoyable, but with some reservations."
        else:
            return "The predicted user score suggests a lower level of satisfaction. Players might not find the game very enjoyable."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted User Score:")
    st.write(interpret_user_score_prediction(predicted_user_score[0]))
    st.write('Number predicted: ',predicted_user_score[0].round(2))

    # # Scatter plot
    # scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    # scatter_ax.scatter(price_df_no_outliers['price'], price_df_no_outliers['user_score'])
    # scatter_ax.set_xlim((0, 100))
    # scatter_ax.set_title('Scatter Plot of Price vs. User Score')
    # scatter_ax.set_xlabel('Price')
    # scatter_ax.set_ylabel('User Score')

    # # Add a point for the user input and predicted value
    # scatter_ax.scatter([game_price], [user_score], color='blue', marker='o', label='User Input')
    # scatter_ax.scatter([game_price], [predicted_user_score[0]], color='red', marker='x', label='Prediction')
    # scatter_ax.legend()

    # # Display the plot
    # st.subheader("Scatter Plot with User Input and Prediction")
    # st.pyplot(scatter_fig)
##################################################################################################################################
# Categorization Impact on Popularity
if st.sidebar.checkbox('Hypothesis 7: Categorization Impact on Popularity', False):
    st.header("7. Categorization Impact on Popularity:")
    st.markdown("- *Theory*: Owner estimates of games tend to be higher for those with more categories and genres.")
    st.markdown('''- *Null Hypothesis*: The number of categories/genres and estimated owners do not significantly correlate.
    - *Alternative Hypothesis*: The estimated owners of games tend to be higher when there are more categories and genres.''')

    category_df =game_data[['appid', 'estimated_owners', 'categories', 'genres']].copy()

    # Create a function to count the number of categories and genres
    def count_categories_genres(row):
        # Split the 'categories' and 'genres' strings into lists
        categories_list = row['categories'].split(',') if not pd.isnull(row['categories']) else []
        genres_list = row['genres'].split(',') if not pd.isnull(row['genres']) else []
        
        # Count the number of unique categories and genres
        num_categories = len(set(categories_list))
        num_genres = len(set(genres_list))
        
        # Return the total count
        return num_categories + num_genres

    # Apply the function to create the new column
    category_df['num_categories_genres'] = category_df.apply(count_categories_genres, axis=1)

    # Scatter plot and Statistical Analysis
    st.header("Categories/Genres vs. Estimated Owners Analysis")

    # Scatter plot
    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='num_categories_genres', y='estimated_owners', data=category_df, ax=scatter_ax)
    scatter_ax.set_title('Scatter Plot of Categories/Genres vs. Estimated Owners')
    scatter_ax.set_xlabel('Number of Categories/Genres')
    scatter_ax.set_ylabel('Estimated Owners')
    st.pyplot(scatter_fig)

    # Detect and handle outliers
    Q1 = category_df['num_categories_genres'].quantile(0.25)
    Q3 = category_df['num_categories_genres'].quantile(0.75)
    IQR = Q3 - Q1

    category_df_no_outliers = category_df[(category_df['num_categories_genres'] < Q1 - 1.5 * IQR) | (category_df['num_categories_genres'] > Q3 + 1.5 * IQR)]

    # Scatter plot and Statistical Analysis
    st.header("Categories/Genres vs. Estimated Owners Analysis (No Outliers)")

    # Scatter plot
    no_scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='num_categories_genres', y='estimated_owners', data=category_df_no_outliers, ax=scatter_ax)
    scatter_ax.set_title('Scatter Plot of Categories/Genres vs. Estimated Owners')
    scatter_ax.set_xlabel('Number of Categories/Genres')
    scatter_ax.set_ylabel('Estimated Owners')
    st.pyplot(no_scatter_fig)
    
    # Calculate the Correlation Coefficient and the P_Values
    st.subheader("Pearson Correlation Coefficient and Hypothesis Testing")
    correlation, p_value = pearsonr(category_df_no_outliers['num_categories_genres'], category_df_no_outliers['estimated_owners'])
    st.write(f'The Pearson Correlation Coefficient is: {correlation}')
    st.write(f'The Pearson P-Value is: {p_value}')

    # Check for statistical significance
    st.subheader("Hypothesis Testing")
    if p_value < 0.05:
        st.write("Dismiss the null hypothesis. There's a noteworthy association.")
    else:
        st.write("Reject the null hypothesis ineffectively. Not a very strong link.")
        
    # Load the pickled model
    model_path = '../data/model/hypothesis7'
    category_model = load_model(model_path)

    st.title("Predicted Model Data")
    st.write(predict_model(category_model, data=category_df))

    # User input for the model prediction
    st.header("Categories/Genres vs. Estimated Owners Prediction")
    st.subheader("Enter Game Information:")
    appid_input = st.number_input("AppID", min_value=0, value=100000, key="appid_input")
    estimated_owners_input = st.number_input("Estimated Owners", min_value=0, value=100000, key="estimated_owners_input")
    num_categories_genres_input = st.number_input("Number of Categories/Genres", min_value=0, value=5, step=1, key="num_categories_genres_input")
    categories_input = st.text_input("Categories (comma-separated)", key="categories_input")
    genres_input = st.text_input("Genres (comma-separated)", key="genres_input")

    # Ensure order and names match the model's expectations
    user_input = pd.DataFrame({
        'appid': [appid_input],
        'estimated_owners': [estimated_owners_input],
        'categories': [categories_input],
        'genres': [genres_input],
        'num_categories_genres': [num_categories_genres_input]
        # Add more columns based on the features used in your model
    })

    # Display user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions using the loaded model
    predicted_owners = category_model.predict(user_input[['categories', 'genres','num_categories_genres']])

    # Function to interpret predicted estimated owners
    def interpret_estimated_owners_prediction(predicted_value):
        if predicted_value >= 50000:
            return "The predicted estimated owners suggest a high level of popularity. The game is likely to have a large player base."
        elif 10000 <= predicted_value < 50000:
            return "The predicted estimated owners indicate a moderate level of popularity. The game may have a decent player base."
        else:
            return "The predicted estimated owners suggest a lower level of popularity. The game might have a smaller player base."

    # Display the interpretation instead of the raw number
    st.subheader("Interpretation of Predicted Estimated Owners:")
    st.write(interpret_estimated_owners_prediction(predicted_owners[0]))
    st.write('Number predicted: ',predicted_owners[0].round(2))

    # # Scatter plot
    # scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    # sns.scatterplot(x='num_categories_genres', y='estimated_owners', data=category_df_no_outliers, ax=scatter_ax)
    # scatter_ax.set_title('Scatter Plot of Categories/Genres vs. Estimated Owners')
    # scatter_ax.set_xlabel('Number of Categories/Genres')
    # scatter_ax.set_ylabel('Estimated Owners')

    # # Add a point for the user input and predicted value
    # scatter_ax.scatter([num_categories_genres_input], [estimated_owners_input], color='red', marker='x', label='User Input')
    # scatter_ax.scatter([num_categories_genres_input], [predicted_owners[0]], color='green', marker='o', label='Prediction')
    # scatter_ax.legend()

    # # Display the plot
    # st.subheader("Scatter Plot with User Input and Prediction")
    # st.pyplot(scatter_fig)
##################################################################################################################################
# Tableau Public URL
tableau_url = """
<div class='tableauPlaceholder' id='viz1707229569152' style='position: relative'><noscript><a href='#'><img alt='Games ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ga&#47;Games_tableau&#47;Games&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Games_tableau&#47;Games' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ga&#47;Games_tableau&#47;Games&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1707229569152');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1177px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
"""

# Header for the Tableau Public link section
st.header("Tableau Public Visualization")

# Embed the Tableau Public link using an IFrame
st.components.v1.html(tableau_url, height=845, width= 1020, scrolling=True)