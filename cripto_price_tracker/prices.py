
import schedule
import time
import threading
import logging
import json
import sqlite3
import requests
import smtplib
import signal
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path


class Configuration:
    """Manages application configuration"""
    
    DEFAULT_CONFIG = {
        "database": {
            "name": "prices.db"
        },
        "api": {
            "coins": ["bitcoin", "ethereum"],
            "currency": "usd",
            "timeout": 10,
            "retry_attempts": 3,
            "retry_delay": 5
        },
        "schedule": {
            "interval_minutes": 5
        },
        "alerts": {
            "threshold_percent": 5.0,
            "email_enabled": False,
            "log_enabled": True
        },
        "email": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender": "",
            "password": "",
            "recipients": []
        },
        "logging": {
            "level": "INFO",
            "file": "crypto_tracker.log",
            "console": True
        }
    }
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    return self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            except json.JSONDecodeError as e:
                print(f"⚠️  Config file error: {e}. Using defaults.")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            self.save_config(self.DEFAULT_CONFIG)
            print(f"✅ Created default config file: {self.config_file}")
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except IOError as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def _merge_configs(self, default, loaded):
        """Recursively merge configurations"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, *keys):
        """Get nested config value"""
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value


class EmailAlert:
    """Handles email notifications"""
    
    def __init__(self, config):
        self.smtp_server = config.get('email', 'smtp_server')
        self.smtp_port = config.get('email', 'smtp_port')
        self.sender = config.get('email', 'sender')
        self.password = config.get('email', 'password')
        self.recipients = config.get('email', 'recipients')
        self.enabled = config.get('alerts', 'email_enabled')
    
    def send_alert(self, subject, message, html=False):
        """Send email alert"""
        if not self.enabled:
            return False
        
        if not self.sender or not self.password or not self.recipients:
            logging.warning("Email not configured properly")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[Crypto Alert] {subject}"
            
            content_type = 'html' if html else 'plain'
            msg.attach(MIMEText(message, content_type))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)
            
            logging.info(f"Email sent: {subject}")
            return True
            
        except smtplib.SMTPException as e:
            logging.error(f"Email SMTP error: {e}")
            return False
        except Exception as e:
            logging.error(f"Email error: {e}")
            return False


class Database:
    """Handles database operations"""
    
    def __init__(self, db_name):
        self.db_name = db_name
        self.setup_database()
    
    def get_connection(self):
        """Create database connection"""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")
            return None
    
    def setup_database(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    change_24h REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    previous_price REAL,
                    current_price REAL,
                    change_percent REAL,
                    timestamp DATETIME NOT NULL,
                    notified BOOLEAN DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_coin_timestamp 
                ON price_history(coin, timestamp)
            ''')
            
            conn.commit()
            logging.info("Database setup complete")
            
        except sqlite3.Error as e:
            logging.error(f"Database setup error: {e}")
        finally:
            conn.close()
    
    def insert_price(self, coin, price, change_24h=None):
        """Insert price record"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
                INSERT INTO price_history (coin, price, timestamp, change_24h)
                VALUES (?, ?, ?, ?)
            ''', (coin, price, timestamp, change_24h))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Insert error: {e}")
            return False
        finally:
            conn.close()
    
    def insert_alert(self, coin, alert_type, prev_price, curr_price, change_pct):
        """Insert alert record"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
                INSERT INTO alerts (coin, alert_type, previous_price, 
                                   current_price, change_percent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (coin, alert_type, prev_price, curr_price, change_pct, timestamp))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Alert insert error: {e}")
            return False
        finally:
            conn.close()
    
    def get_last_price(self, coin):
        """Get last recorded price for a coin"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT price FROM price_history
                WHERE coin = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (coin,))
            
            row = cursor.fetchone()
            return row['price'] if row else None
            
        except sqlite3.Error as e:
            logging.error(f"Query error: {e}")
            return None
        finally:
            conn.close()


class AutomatedCryptoTracker:
    """
    Main tracker class that orchestrates all components
    
    Features:
    - Scheduled price fetching
    - Price spike detection
    - Email and log alerts
    - Graceful start/stop
    - Signal handling
    """
    
    def __init__(self, config_file='config.json'):
        # Load configuration
        self.config = Configuration(config_file)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.db = Database(self.config.get('database', 'name'))
        self.email_alert = EmailAlert(self.config)
        
        # State management
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.last_prices = {}
        self.fetch_count = 0
        self.error_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info("Tracker initialized")
    
    def setup_logging(self):
        """Configure logging system"""
        log_config = self.config.get('logging')
        
        level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'crypto_tracker.log')
        console = log_config.get('console', True)
        
        handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            handlers.append(console_handler)
        
        logging.basicConfig(
            level=level,
            handlers=handlers
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n⚠️  Shutdown signal received...")
        logging.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the automated tracker"""
        with self.lock:
            if self.running:
                print("⚠️  Tracker is already running!")
                return False
            
            self.running = True
        
        # Start worker thread
        self.thread = threading.Thread(target=self._run, daemon=False)
        self.thread.start()
        
        print("✅ Tracker started successfully!")
        print(f"   Fetching prices every {self.config.get('schedule', 'interval_minutes')} minutes")
        print(f"   Alert threshold: {self.config.get('alerts', 'threshold_percent')}%")
        print(f"   Tracking: {', '.join(self.config.get('api', 'coins'))}")
        print("\n⏸️  Press Ctrl+C to stop\n")
        
        logging.info("Tracker started")
        
        return True
    
    def stop(self):
        """Stop the tracker gracefully"""
        with self.lock:
            if not self.running:
                return False
            
            self.running = False
        
        print("\n⏳ Stopping tracker...")
        logging.info("Stopping tracker")
        
        # Clear scheduled tasks
        schedule.clear()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        
        print("✅ Tracker stopped!")
        print(f"   Total fetches: {self.fetch_count}")
        print(f"   Total errors: {self.error_count}")
        
        logging.info(f"Tracker stopped (fetches: {self.fetch_count}, errors: {self.error_count})")
        
        return True
    
    def _run(self):
        """Main worker loop"""
        # Schedule the job
        interval = self.config.get('schedule', 'interval_minutes')
        schedule.every(interval).minutes.do(self.fetch_and_process)
        
        # Run immediately on start
        self.fetch_and_process()
        
        # Main loop
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def fetch_and_process(self):
        """Fetch prices and process alerts"""
        try:
            self.fetch_count += 1
            
            # Fetch prices with retry logic
            data = self._fetch_with_retry()
            
            if not data:
                self.error_count += 1
                return
            
            # Process each coin
            for coin, info in data.items():
                price = info['usd']
                change_24h = info.get('usd_24h_change')
                
                # Save to database
                self.db.insert_price(coin, price, change_24h)
                
                # Check for alerts
                self._check_and_alert(coin, price)
                
                # Update last price
                with self.lock:
                    self.last_prices[coin] = price
            
            logging.info(f"Fetch #{self.fetch_count} completed successfully")
            self._print_status(data)
            
        except Exception as e:
            self.error_count += 1
            logging.error(f"Fetch error: {e}", exc_info=True)
    
    def _fetch_with_retry(self):
        """Fetch prices with retry logic"""
        api_config = self.config.get('api')
        attempts = api_config.get('retry_attempts', 3)
        delay = api_config.get('retry_delay', 5)
        
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        'ids': ','.join(api_config.get('coins')),
                        'vs_currencies': api_config.get('currency'),
                        'include_24hr_change': 'true'
                    },
                    timeout=api_config.get('timeout', 10)
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logging.warning(f"API returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Attempt {attempt}/{attempts} failed: {e}")
                
                if attempt < attempts:
                    time.sleep(delay)
        
        logging.error("All retry attempts failed")
        return None
    
    def _check_and_alert(self, coin, current_price):
        """Check if price change warrants an alert"""
        threshold = self.config.get('alerts', 'threshold_percent')
        
        # Get previous price
        previous_price = self.db.get_last_price(coin)
        
        if previous_price is None:
            return
        
        # Calculate change
        change_percent = ((current_price - previous_price) / previous_price) * 100
        
        if abs(change_percent) >= threshold:
            # Determine alert type
            alert_type = "SPIKE_UP" if change_percent > 0 else "SPIKE_DOWN"
            direction = "increased" if change_percent > 0 else "decreased"
            emoji = "📈" if change_percent > 0 else "📉"
            
            # Log alert
            if self.config.get('alerts', 'log_enabled'):
                logging.warning(
                    f"ALERT: {coin.upper()} {direction} {abs(change_percent):.2f}% "
                    f"(${previous_price:,.2f} → ${current_price:,.2f})"
                )
            
            # Print to console
            print(f"\n🚨 PRICE ALERT!")
            print(f"   Coin: {coin.upper()}")
            print(f"   Change: {emoji} {change_percent:+.2f}%")
            print(f"   From: ${previous_price:,.2f}")
            print(f"   To: ${current_price:,.2f}")
            print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Save alert to database
            self.db.insert_alert(coin, alert_type, previous_price, current_price, change_percent)
            
            # Send email if enabled
            if self.config.get('alerts', 'email_enabled'):
                subject = f"{coin.upper()} {direction} {abs(change_percent):.2f}%"
                message = f"""
Cryptocurrency Price Alert

Coin: {coin.upper()}
Change: {change_percent:+.2f}%
Previous Price: ${previous_price:,.2f}
Current Price: ${current_price:,.2f}
Direction: {direction.upper()}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from your Crypto Tracker.
                """
                self.email_alert.send_alert(subject, message)
    
    def _print_status(self, data):
        """Print current status"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status Update:")
        for coin, info in data.items():
            change = info.get('usd_24h_change', 0) or 0
            arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"  {coin.capitalize():12} ${info['usd']:>12,.2f}  {arrow} {change:>+6.2f}%")
    
    def get_status(self):
        """Get current tracker status"""
        return {
            'running': self.running,
            'fetch_count': self.fetch_count,
            'error_count': self.error_count,
            'last_prices': self.last_prices.copy(),
            'config': self.config.config
        }


def main():
    """Main entry point"""
    print("="*60)
    print("🚀 Automated Cryptocurrency Price Tracker")
    print("="*60)
    print()
    
    # Create tracker instance
    tracker = AutomatedCryptoTracker('config.json')
    
    # Start tracking
    tracker.start()
    
    # Keep main thread alive
    try:
        while tracker.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()