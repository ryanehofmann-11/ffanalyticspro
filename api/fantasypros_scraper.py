# fantasypros_scraper.py

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

def scrape_rankings(position='RB', scoring_format='Standard'):
    """
    Scrape FantasyPros rankings for a specific position and scoring format.
    
    Args:
        position (str): Position to scrape ('QB', 'RB', 'WR', 'TE', 'K', 'DST')
        scoring_format (str): Scoring format ('Standard', 'PPR', 'Half-PPR')
    
    Returns:
        pandas.DataFrame: Rankings data with projections
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)

    try:
        # Construct URL based on scoring format
        base_url = "https://www.fantasypros.com/nfl/rankings/"
        
        # Map scoring formats to URL suffixes
        format_suffix = {
            'Standard': '',
            'PPR': 'ppr-',
            'Half-PPR': 'half-point-ppr-'
        }
        
        if scoring_format in format_suffix:
            url = f"{base_url}{format_suffix[scoring_format]}{position.lower()}.php"
        else:
            print(f"⚠️ Unknown scoring format: {scoring_format}. Using Standard.")
            url = f"{base_url}{position.lower()}.php"
        
        print(f"🌐 Fetching: {url}")
        driver.get(url)
        time.sleep(3)
        
        # Wait for rankings table to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'tr.player-row')))

        # Extract player data
        rows = driver.find_elements(By.CSS_SELECTOR, 'tr.player-row')
        data = []

        for row in rows:
            try:
                # Player name
                name_elem = row.find_element(By.CSS_SELECTOR, 'a.player-cell-name')
                name = name_elem.text.strip()

                # Team
                team_elem = row.find_element(By.CSS_SELECTOR, 'span.player-cell-team')
                team = team_elem.text.strip("()")

                # Opponent
                opp_elem = row.find_element(By.CSS_SELECTOR, 'td.opp-cell')
                opponent = opp_elem.text.strip()

                # Projected fantasy points (last column)
                proj_fpts_elem = row.find_elements(By.CSS_SELECTOR, 'td')[-1]
                proj_fpts = float(proj_fpts_elem.text.strip())

                # Rank
                rank_elem = row.find_element(By.CSS_SELECTOR, 'td.sticky-cell.sticky-cell-one')
                rank = int(rank_elem.text.strip())

                # Start/Sit grade
                start_sit_elem = row.find_element(By.CSS_SELECTOR, 'div.grade-wrap')
                start_sit = start_sit_elem.text.strip()

                data.append({
                    'name': name,
                    'team': team,
                    'opponent': opponent,
                    'proj_fpts': proj_fpts,
                    'rank': rank,
                    'start_sit': start_sit,
                    'position': position,
                    'scoring_format': scoring_format
                })

            except Exception as e:
                print(f"⚠️ Skipping row due to error: {e}")
                continue

        print(f"✅ Successfully scraped {len(data)} {position} players for {scoring_format} scoring")
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return pd.DataFrame()
        
    finally:
        driver.quit()

    df = pd.DataFrame(data)
    return df


def scrape_all_positions(scoring_format='Standard'):
    """
    Scrape rankings for all fantasy-relevant positions.
    
    Args:
        scoring_format (str): Scoring format ('Standard', 'PPR', 'Half-PPR')
    
    Returns:
        dict: Dictionary with position as key and DataFrame as value
    """
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    all_data = {}
    
    for position in positions:
        print(f"🔄 Scraping {position} rankings...")
        df = scrape_rankings(position, scoring_format)
        if not df.empty:
            all_data[position] = df
        else:
            print(f"⚠️ No data retrieved for {position}")
    
    return all_data
