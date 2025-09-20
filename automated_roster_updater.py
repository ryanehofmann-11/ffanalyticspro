#!/usr/bin/env python3
"""
Automated Roster Updater - Real-time NFL Roster Data
This automatically fetches current roster information from multiple reliable sources.
"""

import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class AutomatedRosterUpdater:
    def __init__(self):
        """Initialize the automated roster updater."""
        self.current_season = 2025
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Data sources
        self.data_sources = {
            'espn': 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/2025/segments/0/leagues/1/players',
            'sleeper': 'https://api.sleeper.app/v1/players/nfl',
            'nfl': 'https://api.nfl.com/v1/players',
            'pff': 'https://api.pff.com/v1/players'
        }
        
        # Team mappings
        self.team_mappings = {
            'ARI': 'Arizona Cardinals',
            'ATL': 'Atlanta Falcons',
            'BAL': 'Baltimore Ravens',
            'BUF': 'Buffalo Bills',
            'CAR': 'Carolina Panthers',
            'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals',
            'CLE': 'Cleveland Browns',
            'DAL': 'Dallas Cowboys',
            'DEN': 'Denver Broncos',
            'DET': 'Detroit Lions',
            'GB': 'Green Bay Packers',
            'HOU': 'Houston Texans',
            'IND': 'Indianapolis Colts',
            'JAX': 'Jacksonville Jaguars',
            'KC': 'Kansas City Chiefs',
            'LAC': 'Los Angeles Chargers',
            'LAR': 'Los Angeles Rams',
            'LV': 'Las Vegas Raiders',
            'MIA': 'Miami Dolphins',
            'MIN': 'Minnesota Vikings',
            'NE': 'New England Patriots',
            'NO': 'New Orleans Saints',
            'NYG': 'New York Giants',
            'NYJ': 'New York Jets',
            'PHI': 'Philadelphia Eagles',
            'PIT': 'Pittsburgh Steelers',
            'SF': 'San Francisco 49ers',
            'SEA': 'Seattle Seahawks',
            'TB': 'Tampa Bay Buccaneers',
            'TEN': 'Tennessee Titans',
            'WAS': 'Washington Commanders'
        }
        
        # Cache for roster data
        self.roster_cache = {}
        self.cache_timestamp = None

    def fetch_sleeper_roster_data(self) -> Dict:
        """Fetch roster data from Sleeper API."""
        try:
            logger.info("Fetching roster data from Sleeper API...")
            url = self.data_sources['sleeper']
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched {len(data)} players from Sleeper")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Sleeper data: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching Sleeper data: {e}")
            return {}

    def fetch_espn_roster_data(self) -> Dict:
        """Fetch roster data from ESPN API."""
        try:
            logger.info("Fetching roster data from ESPN API...")
            url = self.data_sources['espn']
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched ESPN roster data")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ESPN data: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching ESPN data: {e}")
            return {}

    def parse_sleeper_data(self, data: Dict) -> Dict:
        """Parse Sleeper API data into standardized format."""
        parsed_data = {}
        
        for player_id, player_info in data.items():
            if not isinstance(player_info, dict):
                continue
                
            name = player_info.get('full_name', '')
            team = player_info.get('team', '')
            position = player_info.get('position', '')
            status = player_info.get('status', '')
            
            if name and team and position:
                parsed_data[name] = {
                    'team': team,
                    'position': position,
                    'status': status,
                    'player_id': player_id,
                    'source': 'sleeper'
                }
        
        return parsed_data

    def get_player_current_team(self, player_name: str) -> Optional[str]:
        """Get the current team for a specific player."""
        # Check cache first
        if self.roster_cache and self.cache_timestamp:
            cache_age = datetime.now() - self.cache_timestamp
            if cache_age.total_seconds() < 3600:  # Cache valid for 1 hour
                if player_name in self.roster_cache:
                    return self.roster_cache[player_name]['team']
        
        # Fetch fresh data
        sleeper_data = self.fetch_sleeper_roster_data()
        parsed_data = self.parse_sleeper_data(sleeper_data)
        
        # Update cache
        self.roster_cache = parsed_data
        self.cache_timestamp = datetime.now()
        
        # Return team for specific player
        if player_name in parsed_data:
            return parsed_data[player_name]['team']
        
        return None

    def get_current_qbs(self) -> Dict[str, str]:
        """Get current starting QBs for all teams."""
        qbs = {}
        
        # Fetch roster data
        sleeper_data = self.fetch_sleeper_roster_data()
        parsed_data = self.parse_sleeper_data(sleeper_data)
        
        # Find QBs
        for player_name, player_info in parsed_data.items():
            if player_info['position'] == 'QB' and player_info['status'] == 'Active':
                team = player_info['team']
                if team not in qbs:  # Take first active QB per team
                    qbs[team] = player_name
        
        return qbs

    def validate_player_roster(self, player_name: str, expected_team: str, position: str) -> Dict:
        """Validate if a player is on the expected team."""
        logger.info(f"Validating {player_name} ({position}, {expected_team})...")
        
        # Get current team
        current_team = self.get_player_current_team(player_name)
        
        result = {
            'player_name': player_name,
            'expected_team': expected_team,
            'current_team': current_team,
            'position': position,
            'is_correct': current_team == expected_team,
            'validation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if result['is_correct']:
            logger.info(f"✅ {player_name} is correctly on {current_team}")
        else:
            logger.warning(f"❌ {player_name} roster mismatch: expected {expected_team}, found {current_team}")
        
        return result

    def validate_fantasy_team(self, players: List[Dict]) -> Dict:
        """Validate an entire fantasy team roster."""
        logger.info(f"🔍 VALIDATING FANTASY TEAM ROSTER - {self.current_season} SEASON")
        logger.info(f"Last Updated: {self.last_updated}")
        logger.info("=" * 60)
        
        validation_results = []
        all_correct = True
        
        for player in players:
            name = player['name']
            team = player['team']
            position = player['position']
            
            result = self.validate_player_roster(name, team, position)
            validation_results.append(result)
            
            if not result['is_correct']:
                all_correct = False
        
        # Summary
        logger.info(f"\n📋 VALIDATION SUMMARY:")
        logger.info("-" * 30)
        if all_correct:
            logger.info("✅ All roster information is up-to-date!")
        else:
            logger.info("❌ Found roster issues that need attention")
            for result in validation_results:
                if not result['is_correct']:
                    logger.info(f"  {result['player_name']}: expected {result['expected_team']}, found {result['current_team']}")
        
        return {
            'all_correct': all_correct,
            'results': validation_results,
            'validation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_roster_summary(self) -> Dict:
        """Get a comprehensive roster summary."""
        logger.info(f"\n📊 {self.current_season} SEASON ROSTER SUMMARY")
        logger.info("=" * 50)
        
        # Get current QBs
        current_qbs = self.get_current_qbs()
        
        logger.info(f"\n🏈 CURRENT STARTING QBs:")
        for team, qb in current_qbs.items():
            team_name = self.team_mappings.get(team, team)
            logger.info(f"  {team} ({team_name}): {qb}")
        
        return {
            'current_qbs': current_qbs,
            'team_mappings': self.team_mappings,
            'last_updated': self.last_updated
        }

def validate_your_team_automated():
    """Validate your specific fantasy team roster using automated data."""
    updater = AutomatedRosterUpdater()
    
    # Your current starters from the screenshots
    current_starters = [
        {"name": "Lamar Jackson", "position": "QB", "team": "BAL", "opponent": "DET", "current_proj": 23.6},
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL", "opponent": "CAR", "current_proj": 21.0},
        {"name": "Jonathan Taylor", "position": "RB", "team": "IND", "opponent": "TEN", "current_proj": 18.7},
        {"name": "Cooper Kupp", "position": "WR", "team": "SEA", "opponent": "NO", "current_proj": 10.5},
        {"name": "Ricky Pearsall", "position": "WR", "team": "SF", "opponent": "ARI", "current_proj": 11.3},
        {"name": "Tyler Warren", "position": "TE", "team": "IND", "opponent": "TEN", "current_proj": 11.6},
        {"name": "Jordan Mason", "position": "RB", "team": "MIN", "opponent": "CIN", "current_proj": 14.7}
    ]
    
    # Your bench players from the screenshots
    bench_players = [
        {"name": "Xavier Worthy", "position": "WR", "team": "KC", "opponent": "NYG", "current_proj": 12.5},
        {"name": "DJ Moore", "position": "WR", "team": "CHI", "opponent": "DAL", "current_proj": 12.9},
        {"name": "Isiah Pacheco", "position": "RB", "team": "KC", "opponent": "NYG", "current_proj": 11.0},
        {"name": "Kyle Pitts", "position": "TE", "team": "ATL", "opponent": "CAR", "current_proj": 9.5},
        {"name": "Wan'Dale Robinson", "position": "WR", "team": "NYG", "opponent": "KC", "current_proj": 11.6},
        {"name": "Michael Penix Jr.", "position": "QB", "team": "ATL", "opponent": "CAR", "current_proj": 16.2},
        {"name": "Troy Franklin", "position": "WR", "team": "DEN", "opponent": "LAC", "current_proj": 11.0}
    ]
    
    all_players = current_starters + bench_players
    
    # Validate roster
    validation_result = updater.validate_fantasy_team(all_players)
    
    # Show roster summary
    updater.get_roster_summary()
    
    return validation_result

if __name__ == "__main__":
    validate_your_team_automated()
