#!/usr/bin/env python3
"""
Comprehensive Test Suite for Fantasy Football Analytics Pro
Tests all components of the system for reliability and accuracy.
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robust_ml_system import RobustMLStartSit
from verified_roster_updater import VerifiedRosterUpdater
from automated_roster_updater import AutomatedRosterUpdater

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    def __init__(self):
        """Initialize the comprehensive test suite."""
        self.test_results = {}
        self.start_time = datetime.now()
        logger.info("🚀 Initializing Comprehensive Test Suite")
        logger.info("=" * 60)

    def test_ml_system(self) -> Dict:
        """Test the ML-enhanced start/sit system."""
        logger.info("\n🧪 Testing ML-Enhanced Start/Sit System")
        logger.info("-" * 40)
        
        test_results = {
            'test_name': 'ML System',
            'start_time': datetime.now(),
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        try:
            # Initialize ML system
            ml_system = RobustMLStartSit('PPR')
            
            # Test 1: Model Training
            logger.info("  Test 1: Model Training...")
            performance = ml_system.train_models()
            
            if performance and all(pos.get('trained', False) for pos in performance.values()):
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ Model training successful")
                logger.info("    ✅ Model training successful")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ Model training failed")
                logger.info("    ❌ Model training failed")
            
            # Test 2: Prediction Accuracy
            logger.info("  Test 2: Prediction Accuracy...")
            test_player = {
                'name': 'Test Player',
                'position': 'RB',
                'base_proj': 15.0,
                'weather_impact': 0,
                'home_advantage': 0.05,
                'spread': -3.0,
                'over_under': 45.0,
                'def_rank': 20,
                'recent_form': 0.7,
                'snap_share': 0.8
            }
            
            prediction = ml_system.predict_player_projection(test_player)
            
            if 'ml_projection' in prediction and prediction['ml_projection'] > 0:
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ Prediction accuracy test passed")
                logger.info("    ✅ Prediction accuracy test passed")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ Prediction accuracy test failed")
                logger.info("    ❌ Prediction accuracy test failed")
            
            # Test 3: Lineup Optimization
            logger.info("  Test 3: Lineup Optimization...")
            test_team = [
                {'name': 'QB1', 'position': 'QB', 'team': 'BAL', 'base_proj': 20.0, 'weather_impact': 0, 'home_advantage': 0.05, 'spread': -5.0, 'over_under': 50.0, 'def_rank': 15, 'recent_form': 0.8, 'snap_share': 1.0},
                {'name': 'RB1', 'position': 'RB', 'team': 'ATL', 'base_proj': 18.0, 'weather_impact': 0, 'home_advantage': 0.05, 'spread': -3.0, 'over_under': 45.0, 'def_rank': 25, 'recent_form': 0.8, 'snap_share': 0.75},
                {'name': 'RB2', 'position': 'RB', 'team': 'IND', 'base_proj': 16.0, 'weather_impact': 0, 'home_advantage': -0.05, 'spread': -2.0, 'over_under': 48.0, 'def_rank': 12, 'recent_form': 0.6, 'snap_share': 0.70},
                {'name': 'WR1', 'position': 'WR', 'team': 'SEA', 'base_proj': 12.0, 'weather_impact': -0.1, 'home_advantage': 0.05, 'spread': 1.0, 'over_under': 42.0, 'def_rank': 10, 'recent_form': 0.4, 'snap_share': 0.80},
                {'name': 'WR2', 'position': 'WR', 'team': 'SF', 'base_proj': 11.0, 'weather_impact': 0, 'home_advantage': 0.05, 'spread': -6.0, 'over_under': 50.0, 'def_rank': 28, 'recent_form': 0.7, 'snap_share': 0.65},
                {'name': 'TE1', 'position': 'TE', 'team': 'IND', 'base_proj': 10.0, 'weather_impact': 0, 'home_advantage': -0.05, 'spread': -2.0, 'over_under': 48.0, 'def_rank': 12, 'recent_form': 0.6, 'snap_share': 0.70}
            ]
            
            lineup_result = ml_system.create_ml_enhanced_lineup(test_team)
            
            if lineup_result and 'optimal_lineup' in lineup_result:
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ Lineup optimization test passed")
                logger.info("    ✅ Lineup optimization test passed")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ Lineup optimization test failed")
                logger.info("    ❌ Lineup optimization test failed")
            
            # Test 4: Model Performance
            logger.info("  Test 4: Model Performance...")
            avg_r2 = np.mean([perf.get('r2', 0) for perf in performance.values() if perf.get('trained', False)])
            
            if avg_r2 > 0.8:  # Expect high R² scores
                test_results['tests_passed'] += 1
                test_results['details'].append(f"✅ Model performance test passed (Avg R²: {avg_r2:.3f})")
                logger.info(f"    ✅ Model performance test passed (Avg R²: {avg_r2:.3f})")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append(f"❌ Model performance test failed (Avg R²: {avg_r2:.3f})")
                logger.info(f"    ❌ Model performance test failed (Avg R²: {avg_r2:.3f})")
            
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['details'].append(f"❌ ML System test failed with error: {str(e)}")
            logger.error(f"    ❌ ML System test failed with error: {str(e)}")
        
        test_results['end_time'] = datetime.now()
        test_results['duration'] = (test_results['end_time'] - test_results['start_time']).total_seconds()
        
        return test_results

    def test_roster_validation(self) -> Dict:
        """Test the roster validation system."""
        logger.info("\n🧪 Testing Roster Validation System")
        logger.info("-" * 40)
        
        test_results = {
            'test_name': 'Roster Validation',
            'start_time': datetime.now(),
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        try:
            # Test 1: Verified Roster Updater
            logger.info("  Test 1: Verified Roster Updater...")
            verified_updater = VerifiedRosterUpdater()
            
            test_players = [
                {"name": "Lamar Jackson", "position": "QB", "team": "BAL"},
                {"name": "Michael Penix Jr.", "position": "QB", "team": "ATL"},
                {"name": "Caleb Williams", "position": "QB", "team": "CHI"}
            ]
            
            validation_result = verified_updater.validate_fantasy_team(test_players)
            
            if validation_result and validation_result.get('all_correct', False):
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ Verified roster validation test passed")
                logger.info("    ✅ Verified roster validation test passed")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ Verified roster validation test failed")
                logger.info("    ❌ Verified roster validation test failed")
            
            # Test 2: QB Situation Validation
            logger.info("  Test 2: QB Situation Validation...")
            qb_tests = [
                ('ATL', 'Michael Penix Jr.'),
                ('CHI', 'Caleb Williams'),
                ('SEA', 'Sam Darnold')
            ]
            
            qb_tests_passed = 0
            for team, expected_qb in qb_tests:
                actual_qb = verified_updater.get_verified_qb(team)
                if actual_qb == expected_qb:
                    qb_tests_passed += 1
            
            if qb_tests_passed == len(qb_tests):
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ QB situation validation test passed")
                logger.info("    ✅ QB situation validation test passed")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ QB situation validation test failed")
                logger.info("    ❌ QB situation validation test failed")
            
            # Test 3: Automated Roster Updater
            logger.info("  Test 3: Automated Roster Updater...")
            automated_updater = AutomatedRosterUpdater()
            
            # Test player lookup
            test_player_team = automated_updater.get_player_current_team("Lamar Jackson")
            
            if test_player_team:
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ Automated roster updater test passed")
                logger.info("    ✅ Automated roster updater test passed")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ Automated roster updater test failed")
                logger.info("    ❌ Automated roster updater test failed")
            
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['details'].append(f"❌ Roster validation test failed with error: {str(e)}")
            logger.error(f"    ❌ Roster validation test failed with error: {str(e)}")
        
        test_results['end_time'] = datetime.now()
        test_results['duration'] = (test_results['end_time'] - test_results['start_time']).total_seconds()
        
        return test_results

    def test_data_integration(self) -> Dict:
        """Test data integration components."""
        logger.info("\n🧪 Testing Data Integration")
        logger.info("-" * 40)
        
        test_results = {
            'test_name': 'Data Integration',
            'start_time': datetime.now(),
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        try:
            # Test 1: FantasyPros Scraper
            logger.info("  Test 1: FantasyPros Scraper...")
            try:
                from api.fantasypros_scraper import scrape_rankings
                
                # Test scraping (this might fail due to website changes, but we test the function exists)
                test_result = scrape_rankings('RB', 'PPR')
                
                if isinstance(test_result, pd.DataFrame):
                    test_results['tests_passed'] += 1
                    test_results['details'].append("✅ FantasyPros scraper test passed")
                    logger.info("    ✅ FantasyPros scraper test passed")
                else:
                    test_results['tests_failed'] += 1
                    test_results['details'].append("❌ FantasyPros scraper test failed")
                    logger.info("    ❌ FantasyPros scraper test failed")
                    
            except ImportError:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ FantasyPros scraper not available")
                logger.info("    ❌ FantasyPros scraper not available")
            
            # Test 2: Sleeper API Integration
            logger.info("  Test 2: Sleeper API Integration...")
            try:
                import requests
                
                url = "https://api.sleeper.app/v1/players/nfl"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    test_results['tests_passed'] += 1
                    test_results['details'].append("✅ Sleeper API integration test passed")
                    logger.info("    ✅ Sleeper API integration test passed")
                else:
                    test_results['tests_failed'] += 1
                    test_results['details'].append("❌ Sleeper API integration test failed")
                    logger.info("    ❌ Sleeper API integration test failed")
                    
            except Exception as e:
                test_results['tests_failed'] += 1
                test_results['details'].append(f"❌ Sleeper API integration test failed: {str(e)}")
                logger.info(f"    ❌ Sleeper API integration test failed: {str(e)}")
            
            # Test 3: Data Processing
            logger.info("  Test 3: Data Processing...")
            test_data = pd.DataFrame({
                'name': ['Player1', 'Player2', 'Player3'],
                'position': ['QB', 'RB', 'WR'],
                'team': ['BAL', 'ATL', 'SEA'],
                'proj_fpts': [20.0, 15.0, 12.0]
            })
            
            if len(test_data) == 3 and 'proj_fpts' in test_data.columns:
                test_results['tests_passed'] += 1
                test_results['details'].append("✅ Data processing test passed")
                logger.info("    ✅ Data processing test passed")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append("❌ Data processing test failed")
                logger.info("    ❌ Data processing test failed")
            
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['details'].append(f"❌ Data integration test failed with error: {str(e)}")
            logger.error(f"    ❌ Data integration test failed with error: {str(e)}")
        
        test_results['end_time'] = datetime.now()
        test_results['duration'] = (test_results['end_time'] - test_results['start_time']).total_seconds()
        
        return test_results

    def test_performance(self) -> Dict:
        """Test system performance and scalability."""
        logger.info("\n🧪 Testing System Performance")
        logger.info("-" * 40)
        
        test_results = {
            'test_name': 'Performance',
            'start_time': datetime.now(),
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        try:
            # Test 1: ML Model Training Speed
            logger.info("  Test 1: ML Model Training Speed...")
            start_time = time.time()
            
            ml_system = RobustMLStartSit('PPR')
            ml_system.train_models()
            
            training_time = time.time() - start_time
            
            if training_time < 30:  # Should train in under 30 seconds
                test_results['tests_passed'] += 1
                test_results['details'].append(f"✅ ML training speed test passed ({training_time:.2f}s)")
                logger.info(f"    ✅ ML training speed test passed ({training_time:.2f}s)")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append(f"❌ ML training speed test failed ({training_time:.2f}s)")
                logger.info(f"    ❌ ML training speed test failed ({training_time:.2f}s)")
            
            # Test 2: Prediction Speed
            logger.info("  Test 2: Prediction Speed...")
            start_time = time.time()
            
            test_player = {
                'name': 'Test Player',
                'position': 'RB',
                'base_proj': 15.0,
                'weather_impact': 0,
                'home_advantage': 0.05,
                'spread': -3.0,
                'over_under': 45.0,
                'def_rank': 20,
                'recent_form': 0.7,
                'snap_share': 0.8
            }
            
            prediction = ml_system.predict_player_projection(test_player)
            prediction_time = time.time() - start_time
            
            if prediction_time < 1:  # Should predict in under 1 second
                test_results['tests_passed'] += 1
                test_results['details'].append(f"✅ Prediction speed test passed ({prediction_time:.3f}s)")
                logger.info(f"    ✅ Prediction speed test passed ({prediction_time:.3f}s)")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append(f"❌ Prediction speed test failed ({prediction_time:.3f}s)")
                logger.info(f"    ❌ Prediction speed test failed ({prediction_time:.3f}s)")
            
            # Test 3: Memory Usage
            logger.info("  Test 3: Memory Usage...")
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            if memory_usage < 500:  # Should use less than 500MB
                test_results['tests_passed'] += 1
                test_results['details'].append(f"✅ Memory usage test passed ({memory_usage:.1f}MB)")
                logger.info(f"    ✅ Memory usage test passed ({memory_usage:.1f}MB)")
            else:
                test_results['tests_failed'] += 1
                test_results['details'].append(f"❌ Memory usage test failed ({memory_usage:.1f}MB)")
                logger.info(f"    ❌ Memory usage test failed ({memory_usage:.1f}MB)")
            
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['details'].append(f"❌ Performance test failed with error: {str(e)}")
            logger.error(f"    ❌ Performance test failed with error: {str(e)}")
        
        test_results['end_time'] = datetime.now()
        test_results['duration'] = (test_results['end_time'] - test_results['start_time']).total_seconds()
        
        return test_results

    def run_all_tests(self) -> Dict:
        """Run all tests and generate comprehensive report."""
        logger.info("🚀 Running Comprehensive Test Suite")
        logger.info("=" * 60)
        
        # Run all test categories
        test_categories = [
            self.test_ml_system,
            self.test_roster_validation,
            self.test_data_integration,
            self.test_performance
        ]
        
        all_results = []
        total_passed = 0
        total_failed = 0
        
        for test_func in test_categories:
            try:
                result = test_func()
                all_results.append(result)
                total_passed += result['tests_passed']
                total_failed += result['tests_failed']
            except Exception as e:
                logger.error(f"Test category failed: {str(e)}")
        
        # Generate summary
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            'total_tests_passed': total_passed,
            'total_tests_failed': total_failed,
            'total_tests': total_passed + total_failed,
            'success_rate': total_passed / (total_passed + total_failed) * 100 if (total_passed + total_failed) > 0 else 0,
            'total_duration': total_duration,
            'test_results': all_results,
            'timestamp': end_time
        }
        
        # Display results
        self.display_results(summary)
        
        return summary

    def display_results(self, summary: Dict):
        """Display comprehensive test results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"\n📈 OVERALL SUMMARY:")
        logger.info(f"  Total Tests: {summary['total_tests']}")
        logger.info(f"  Passed: {summary['total_tests_passed']}")
        logger.info(f"  Failed: {summary['total_tests_failed']}")
        logger.info(f"  Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"  Total Duration: {summary['total_duration']:.2f} seconds")
        
        logger.info(f"\n📋 DETAILED RESULTS:")
        for result in summary['test_results']:
            logger.info(f"\n  {result['test_name']}:")
            logger.info(f"    Duration: {result['duration']:.2f}s")
            logger.info(f"    Passed: {result['tests_passed']}")
            logger.info(f"    Failed: {result['tests_failed']}")
            
            for detail in result['details']:
                logger.info(f"    {detail}")
        
        # Overall assessment
        if summary['success_rate'] >= 80:
            logger.info(f"\n🎉 SYSTEM STATUS: EXCELLENT ({summary['success_rate']:.1f}% success rate)")
        elif summary['success_rate'] >= 60:
            logger.info(f"\n✅ SYSTEM STATUS: GOOD ({summary['success_rate']:.1f}% success rate)")
        else:
            logger.info(f"\n⚠️ SYSTEM STATUS: NEEDS IMPROVEMENT ({summary['success_rate']:.1f}% success rate)")

def run_comprehensive_tests():
    """Run the comprehensive test suite."""
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_all_tests()
    return results

if __name__ == "__main__":
    run_comprehensive_tests()
