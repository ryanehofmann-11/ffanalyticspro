#!/usr/bin/env python3
"""
Robust ML-Enhanced Start/Sit System
Advanced fantasy football analysis with comprehensive machine learning integration.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class RobustMLStartSit:
    def __init__(self, scoring_format='PPR'):
        """Initialize the robust ML-enhanced start/sit system."""
        self.scoring_format = scoring_format
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.is_trained = False
        
        # Initialize models
        self.models = {
            'QB': RandomForestRegressor(n_estimators=50, random_state=42),
            'RB': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'WR': RandomForestRegressor(n_estimators=50, random_state=42),
            'TE': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'K': LinearRegression(),
            'DST': LinearRegression()
        }
        
        # Initialize scalers
        for position in self.models.keys():
            self.scalers[position] = StandardScaler()
        
        logger.info(f"Initialized Robust ML-Enhanced Start/Sit system for {scoring_format} scoring")

    def create_comprehensive_training_data(self) -> pd.DataFrame:
        """Create comprehensive training dataset with extensive historical data."""
        logger.info("Creating comprehensive training dataset...")
        
        # Generate extensive mock historical data
        np.random.seed(42)  # For reproducibility
        
        training_data = []
        
        # Generate data for each position
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        
        for position in positions:
            # Generate 50 samples per position
            for i in range(50):
                # Base projections vary by position
                base_proj_ranges = {
                    'QB': (15, 30), 'RB': (8, 25), 'WR': (5, 20), 
                    'TE': (3, 15), 'K': (5, 12), 'DST': (3, 15)
                }
                
                base_proj = np.random.uniform(*base_proj_ranges[position])
                
                # Generate realistic features
                weather_impact = np.random.choice([0, -0.1, -0.15, -0.2], p=[0.6, 0.2, 0.15, 0.05])
                home_advantage = np.random.choice([0.05, -0.05], p=[0.5, 0.5])
                spread = np.random.uniform(-10, 10)
                over_under = np.random.uniform(35, 55)
                def_rank = np.random.randint(1, 33)
                recent_form = np.random.uniform(0.3, 1.0)
                snap_share = np.random.uniform(0.4, 1.0)
                
                # Calculate actual points with realistic variance
                # Add noise and realistic adjustments
                weather_adj = base_proj * weather_impact
                home_adj = base_proj * home_advantage
                def_adj = base_proj * (0.1 if def_rank > 20 else -0.1 if def_rank < 8 else 0)
                form_adj = base_proj * (recent_form - 0.5) * 0.2
                
                actual = base_proj + weather_adj + home_adj + def_adj + form_adj
                actual += np.random.normal(0, base_proj * 0.15)  # Add noise
                actual = max(0, actual)  # Ensure non-negative
                
                training_data.append({
                    'position': position,
                    'base_proj': base_proj,
                    'actual': actual,
                    'weather_impact': weather_impact,
                    'home_advantage': home_advantage,
                    'spread': spread,
                    'over_under': over_under,
                    'def_rank': def_rank,
                    'recent_form': recent_form,
                    'snap_share': snap_share
                })
        
        df = pd.DataFrame(training_data)
        logger.info(f"Created comprehensive training dataset with {len(df)} samples")
        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for ML training."""
        # Select features for training
        feature_columns = [
            'base_proj', 'weather_impact', 'home_advantage', 'spread', 
            'over_under', 'def_rank', 'recent_form', 'snap_share'
        ]
        
        X = df[feature_columns].values
        y = df['actual'].values
        
        return X, y

    def train_models(self) -> Dict:
        """Train ML models for each position with robust error handling."""
        logger.info("Training robust ML models...")
        
        # Create comprehensive training data
        training_df = self.create_comprehensive_training_data()
        
        # Train models for each position
        for position in self.models.keys():
            logger.info(f"Training {position} model...")
            
            # Filter data for position
            pos_data = training_df[training_df['position'] == position]
            
            if len(pos_data) < 10:  # Need minimum samples
                logger.warning(f"Insufficient data for {position}, skipping training")
                continue
            
            try:
                # Extract features
                X, y = self.extract_features(pos_data)
                
                # Scale features
                X_scaled = self.scalers[position].fit_transform(X)
                
                # Train model
                self.models[position].fit(X_scaled, y)
                
                # Calculate performance metrics
                y_pred = self.models[position].predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                self.model_performance[position] = {
                    'mse': mse,
                    'r2': r2,
                    'samples': len(pos_data),
                    'trained': True
                }
                
                # Get feature importance
                if hasattr(self.models[position], 'feature_importances_'):
                    feature_names = ['base_proj', 'weather_impact', 'home_advantage', 'spread', 
                                   'over_under', 'def_rank', 'recent_form', 'snap_share']
                    self.feature_importance[position] = dict(zip(feature_names, self.models[position].feature_importances_))
                
                logger.info(f"✅ {position} model trained - R²: {r2:.3f}, MSE: {mse:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {position} model: {e}")
                self.model_performance[position] = {
                    'mse': float('inf'),
                    'r2': 0.0,
                    'samples': len(pos_data),
                    'trained': False
                }
        
        self.is_trained = True
        return self.model_performance

    def predict_player_projection(self, player_data: Dict) -> Dict:
        """Predict enhanced projection for a player using ML with fallback."""
        position = player_data['position']
        
        # Check if model is trained and available
        if not self.is_trained or position not in self.model_performance:
            logger.warning(f"No trained model available for {position}, using base projection")
            return {
                **player_data,
                'ml_projection': player_data.get('base_proj', 0),
                'confidence': 0.5,
                'model_used': 'fallback'
            }
        
        if not self.model_performance[position].get('trained', False):
            logger.warning(f"Model for {position} not properly trained, using base projection")
            return {
                **player_data,
                'ml_projection': player_data.get('base_proj', 0),
                'confidence': 0.5,
                'model_used': 'fallback'
            }
        
        try:
            # Prepare features
            features = np.array([[
                player_data.get('base_proj', 0),
                player_data.get('weather_impact', 0),
                player_data.get('home_advantage', 0),
                player_data.get('spread', 0),
                player_data.get('over_under', 45),
                player_data.get('def_rank', 16),
                player_data.get('recent_form', 0.5),
                player_data.get('snap_share', 0.7)
            ]])
            
            # Scale features
            features_scaled = self.scalers[position].transform(features)
            
            # Make prediction
            ml_projection = self.models[position].predict(features_scaled)[0]
            
            # Calculate confidence based on model performance
            confidence = self.model_performance[position].get('r2', 0.5)
            
            return {
                **player_data,
                'ml_projection': ml_projection,
                'confidence': confidence,
                'model_used': type(self.models[position]).__name__
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {position}: {e}")
            return {
                **player_data,
                'ml_projection': player_data.get('base_proj', 0),
                'confidence': 0.5,
                'model_used': 'error_fallback'
            }

    def create_ml_enhanced_lineup(self, players: List[Dict]) -> Dict:
        """Create optimal lineup using ML-enhanced projections."""
        logger.info("Creating ML-enhanced lineup...")
        
        # Train models if not already trained
        if not self.is_trained:
            self.train_models()
        
        # Enhance each player with ML projections
        enhanced_players = []
        for player in players:
            enhanced_player = self.predict_player_projection(player)
            enhanced_players.append(enhanced_player)
        
        # Sort by ML projection
        enhanced_players.sort(key=lambda x: x.get('ml_projection', x.get('base_proj', 0)), reverse=True)
        
        # Create optimal lineup
        optimal_lineup = {
            'QB': [],
            'RB': [],
            'WR': [],
            'TE': [],
            'FLEX': []
        }
        
        # Fill positions
        for player in enhanced_players:
            position = player['position']
            
            if position == 'QB' and len(optimal_lineup['QB']) < 1:
                optimal_lineup['QB'].append(player)
            elif position == 'RB' and len(optimal_lineup['RB']) < 2:
                optimal_lineup['RB'].append(player)
            elif position == 'WR' and len(optimal_lineup['WR']) < 2:
                optimal_lineup['WR'].append(player)
            elif position == 'TE' and len(optimal_lineup['TE']) < 1:
                optimal_lineup['TE'].append(player)
            elif position in ['RB', 'WR', 'TE'] and len(optimal_lineup['FLEX']) < 1:
                optimal_lineup['FLEX'].append(player)
        
        # Calculate total projected points
        total_points = 0
        for position, starters in optimal_lineup.items():
            for starter in starters:
                total_points += starter.get('ml_projection', starter.get('base_proj', 0))
        
        return {
            'optimal_lineup': optimal_lineup,
            'total_projected_points': total_points,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'enhanced_players': enhanced_players
        }

    def analyze_feature_importance(self) -> Dict:
        """Analyze feature importance across all positions."""
        logger.info("Analyzing feature importance...")
        
        analysis = {}
        for position, importance in self.feature_importance.items():
            if importance:
                # Sort features by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                analysis[position] = sorted_features
        
        return analysis

    def save_models(self, filepath: str = 'models/'):
        """Save trained models to disk."""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        for position, model in self.models.items():
            if self.model_performance.get(position, {}).get('trained', False):
                model_path = os.path.join(filepath, f'{position}_model.pkl')
                joblib.dump(model, model_path)
                logger.info(f"Saved {position} model to {model_path}")
        
        # Save scalers
        for position, scaler in self.scalers.items():
            if self.model_performance.get(position, {}).get('trained', False):
                scaler_path = os.path.join(filepath, f'{position}_scaler.pkl')
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved {position} scaler to {scaler_path}")

def test_robust_ml_system():
    """Test the robust ML-enhanced start/sit system."""
    logger.info("🚀 Testing Robust ML-Enhanced Start/Sit System")
    logger.info("=" * 60)
    
    # Initialize system
    ml_system = RobustMLStartSit('PPR')
    
    # Your team data with comprehensive features
    your_team = [
        {"name": "Lamar Jackson", "position": "QB", "team": "BAL", "opponent": "DET", "base_proj": 23.6, "weather_impact": 0, "home_advantage": 0.05, "spread": -7.0, "over_under": 52.0, "def_rank": 15, "recent_form": 0.8, "snap_share": 1.0},
        {"name": "Bijan Robinson", "position": "RB", "team": "ATL", "opponent": "CAR", "base_proj": 21.0, "weather_impact": 0, "home_advantage": 0.05, "spread": -3.0, "over_under": 45.0, "def_rank": 25, "recent_form": 0.8, "snap_share": 0.75},
        {"name": "Jonathan Taylor", "position": "RB", "team": "IND", "opponent": "TEN", "base_proj": 18.7, "weather_impact": 0, "home_advantage": -0.05, "spread": -2.0, "over_under": 48.0, "def_rank": 12, "recent_form": 0.6, "snap_share": 0.70},
        {"name": "Cooper Kupp", "position": "WR", "team": "SEA", "opponent": "NO", "base_proj": 10.5, "weather_impact": -0.1, "home_advantage": 0.05, "spread": 1.0, "over_under": 42.0, "def_rank": 10, "recent_form": 0.4, "snap_share": 0.80},
        {"name": "Ricky Pearsall", "position": "WR", "team": "SF", "opponent": "ARI", "base_proj": 11.3, "weather_impact": 0, "home_advantage": 0.05, "spread": -6.0, "over_under": 50.0, "def_rank": 28, "recent_form": 0.7, "snap_share": 0.65},
        {"name": "Xavier Worthy", "position": "WR", "team": "KC", "opponent": "NYG", "base_proj": 12.5, "weather_impact": 0, "home_advantage": 0.05, "spread": -8.0, "over_under": 54.0, "def_rank": 32, "recent_form": 0.7, "snap_share": 0.65},
        {"name": "Tyler Warren", "position": "TE", "team": "IND", "opponent": "TEN", "base_proj": 11.6, "weather_impact": 0, "home_advantage": -0.05, "spread": -2.0, "over_under": 48.0, "def_rank": 12, "recent_form": 0.6, "snap_share": 0.70},
        {"name": "Jordan Mason", "position": "RB", "team": "MIN", "opponent": "CIN", "base_proj": 14.7, "weather_impact": 0, "home_advantage": 0.05, "spread": -1.0, "over_under": 46.0, "def_rank": 18, "recent_form": 0.5, "snap_share": 0.60}
    ]
    
    # Create ML-enhanced lineup
    result = ml_system.create_ml_enhanced_lineup(your_team)
    
    # Display results
    logger.info(f"\n🏆 ML-ENHANCED OPTIMAL LINEUP:")
    logger.info("=" * 50)
    
    total_points = 0
    for position, starters in result['optimal_lineup'].items():
        if starters:
            logger.info(f"\n{position}:")
            for starter in starters:
                ml_proj = starter.get('ml_projection', starter.get('base_proj', 0))
                base_proj = starter.get('base_proj', 0)
                confidence = starter.get('confidence', 0)
                model_used = starter.get('model_used', 'unknown')
                logger.info(f"  {starter['name']} ({starter['team']}) - ML: {ml_proj:.1f} pts (Base: {base_proj:.1f}, Confidence: {confidence:.2f}, Model: {model_used})")
                total_points += ml_proj
    
    logger.info(f"\n🎯 Total ML-Enhanced Projected Points: {total_points:.1f}")
    
    # Show model performance
    logger.info(f"\n📊 MODEL PERFORMANCE:")
    logger.info("-" * 30)
    for position, performance in result['model_performance'].items():
        if performance.get('trained', False):
            logger.info(f"  {position}: R² = {performance['r2']:.3f}, MSE = {performance['mse']:.3f}, Samples = {performance['samples']}")
        else:
            logger.info(f"  {position}: Not trained")
    
    # Show feature importance
    feature_analysis = ml_system.analyze_feature_importance()
    logger.info(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    logger.info("-" * 40)
    for position, features in feature_analysis.items():
        logger.info(f"\n{position}:")
        for feature, importance in features[:3]:  # Top 3 features
            logger.info(f"  {feature}: {importance:.3f}")
    
    # Save models
    ml_system.save_models()
    
    return result

if __name__ == "__main__":
    test_robust_ml_system()
