-- Create the steam_games schema
CREATE SCHEMA IF NOT EXISTS steam_games;

-- Create tables and add columns
USE steam_games;

-- Create Game Information Table
CREATE TABLE IF NOT EXISTS game (
    appid INT PRIMARY KEY,
    name VARCHAR(1000),
    release_date DATE,
    estimated_owners FLOAT,
    peak_ccu FLOAT,
    required_age FLOAT,
    price FLOAT,
    dlc_count FLOAT,
    -- about_the_game VARCHAR(1000),
    reviews VARCHAR(1000),
    header_image VARCHAR(1000),
    website VARCHAR(1000),
    support_url VARCHAR(1000),
    support_email VARCHAR(1000)
);

-- Create Language Table
CREATE TABLE IF NOT EXISTS languages (
    appid INT PRIMARY KEY,
    supported_languages VARCHAR(1000),
    full_audio_languages VARCHAR(1000),
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create Platform Information Table
CREATE TABLE IF NOT EXISTS platform (
    appid INT PRIMARY KEY,
    windows BOOL,
    mac BOOL,
    linux BOOL,
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create Metacritic Information Table
CREATE TABLE IF NOT EXISTS metacritic (
    appid INT PRIMARY KEY,
    metacritic_score FLOAT,
    metacritic_url VARCHAR(1000),
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create User Feedback Table
CREATE TABLE IF NOT EXISTS user_feedback (
    appid INT PRIMARY KEY,
    user_score FLOAT,
    positive FLOAT,
    negative FLOAT,
    score_rank FLOAT,
    achievements FLOAT,
    recommendations FLOAT,
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create Playtime Information Table
CREATE TABLE IF NOT EXISTS playtime (
    appid INT PRIMARY KEY,
    average_playtime_forever FLOAT,
    average_playtime_two_weeks FLOAT,
    median_playtime_forever FLOAT,
    median_playtime_two_weeks FLOAT,
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create Development Information Table
CREATE TABLE IF NOT EXISTS development (
    appid INT PRIMARY KEY,
    developers VARCHAR(1000),
    publishers VARCHAR(1000),
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create Categorization Tables
CREATE TABLE IF NOT EXISTS categorization (
    appid INT PRIMARY KEY,
    categories VARCHAR(1000),
    genres VARCHAR(1000),
    tags VARCHAR(1000),
    FOREIGN KEY (appid) REFERENCES game(appid)
);

-- Create Media Tables
CREATE TABLE IF NOT EXISTS media (
    appid INT PRIMARY KEY,
    screenshots VARCHAR(1000),
    movies VARCHAR(1000),
    FOREIGN KEY (appid) REFERENCES game(appid)
);
