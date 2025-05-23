# app.py
import streamlit as st
import psycopg2
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
from psycopg2 import OperationalError

# Ładowanie zmiennych środowiskowych
load_dotenv()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=st.secrets.neon.host if 'neon' in st.secrets else os.getenv('DB_HOST'),
            database=st.secrets.neon.database if 'neon' in st.secrets else os.getenv('DB_NAME'),
            user=st.secrets.neon.user if 'neon' in st.secrets else os.getenv('DB_USER'),
            password=st.secrets.neon.password if 'neon' in st.secrets else os.getenv('DB_PASSWORD'),
            port=st.secrets.neon.port if 'neon' in st.secrets else os.getenv('DB_PORT', '5432'),
            sslmode='require',  # Wymagane dla Neon.tech
            connect_timeout=3
        )
        return conn
    except Exception as e:
        st.error(f"""
        🔴 **Błąd połączenia z bazą danych**  
        Upewnij się że:  
        - Dane dostępowe są poprawne  
        - Neon.tech ma whitelistowane Twoje IP  
        - Połączenie używa SSL  
        Błąd: {str(e)}
        """)
        st.stop()

def get_recent_timestamps(limit=10):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT created_at 
                FROM recommendations 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            return [record[0] for record in cur.fetchall()]
    finally:
        conn.close()

def get_recommendations_by_timestamp(selected_timestamp):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM recommendations 
                WHERE created_at = %s
                ORDER BY score DESC, ticker
                """,
                (selected_timestamp,)
            )
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)
    finally:
        conn.close()

def format_recommendation(row):
    color = 'green' if row['recommendation'] == 'BUY' else 'red' if row['recommendation'] == 'SELL' else 'gray'
    return f"<span style='color: {color}; font-weight: bold;'>{row['recommendation']}</span>"

def format_positioning(row):
    color = 'green' if row['positioning'] == 'LONG' else 'red' if row['positioning'] == 'SHORT' else 'gray'
    return f"<span style='color: {color}; font-weight: bold;'>{row['positioning']}</span>"

def format_timestamp_display(timestamp):
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def main():
    st.set_page_config(
        page_title="Stock Recommendations", 
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )
    
    # Dopiero teraz reszta kodu
    st.title("Stock Recommendations Dashboard")
    
    # Test połączenia z bazą
    st.write("🔌 Testing database connection...")
    try:
        test_conn = get_db_connection()
        if test_conn:
            st.success("✅ Database connection successful!")
            test_conn.close()
        else:
            st.stop()
    except Exception as e:
        st.error(f"❌ Connection test failed: {e}")
        st.stop()
        
    # Pobierz 10 najnowszych timestampów
    timestamps = get_recent_timestamps(10)
    
    if not timestamps:
        st.warning("No data available in the database.")
        return

    # Formatowanie timestampów do wyświetlenia
    timestamp_display = [format_timestamp_display(ts) for ts in timestamps]
    
    # Wybór konkretnego timestampu
    selected_display = st.selectbox(
        "Select report time:",
        options=timestamp_display,
        index=0
    )
    
    # Znajdź odpowiadający timestamp
    selected_timestamp = timestamps[timestamp_display.index(selected_display)]

    # Pobierz dane dla wybranego timestampu
    df = get_recommendations_by_timestamp(selected_timestamp)
    
    if df.empty:
        st.warning(f"No data available for {selected_display}")
    else:
        # Formatowanie kolumn
        df['recommendation_formatted'] = df.apply(format_recommendation, axis=1)
        df['positioning_formatted'] = df.apply(format_positioning, axis=1)
        
        # Wybór kolumn do wyświetlenia
        columns_to_show = [
            'ticker', 'price', 'sentiment', 'pe', 'roe', 'de', 'growth',
            'prob_up', 'score', 'recommendation_formatted', 'positioning_formatted',
            'holding_days', 'strong_recommendation', 'watchlist'
        ]
        
        # Filtrowanie - w dwóch kolumnach
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_strong_only = st.checkbox("Only strong", value=False)
            show_watchlist = st.checkbox("Only watchlist", value=False)
            
        with col2:
            # Filtry dla pozycji
            position_filter = st.multiselect(
                "Position type:",
                options=['LONG', 'SHORT', 'NEUTRAL'],
                default=['LONG', 'SHORT', 'NEUTRAL']
            )
            
        with col3:
            # Dodatkowe filtry
            min_score = st.slider(
                "Min score:",
                min_value=-1.0,
                max_value=1.0,
                value=-1.0,
                step=0.1
            )
        
        # Zastosuj filtry
        if show_strong_only:
            df = df[df['strong_recommendation'] == True]
        if show_watchlist:
            df = df[df['watchlist'] == True]
        if position_filter:
            df = df[df['positioning'].isin(position_filter)]
        df = df[df['score'] >= min_score]
        
        # Wyświetl statystyki filtrów
        st.write(f"""
        **Showing:** {len(df)} recommendations | 
        LONG: {len(df[df['positioning'] == 'LONG'])} | 
        SHORT: {len(df[df['positioning'] == 'SHORT'])} | 
        NEUTRAL: {len(df[df['positioning'] == 'NEUTRAL'])}
        """)
        
        # Konwersja DataFrame do HTML z wybranymi kolumnami
        st.write(
            df[columns_to_show].to_html(escape=False, index=False), 
            unsafe_allow_html=True
        )
        
        # Pobierz unikalne tickery dla wykresów
        if not df.empty:
            selected_ticker = st.selectbox(
                "Select ticker to view details:",
                options=sorted(df['ticker'].unique()),
                index=0
            )
            
            # Dodatkowe informacje dla wybranego tickera
            ticker_data = df[df['ticker'] == selected_ticker].iloc[0]
            
            st.subheader(f"Details for {selected_ticker}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Price", f"${ticker_data['price']:.2f}")
                st.metric("Recommendation", ticker_data['recommendation'])
                st.metric("Position", ticker_data['positioning'])
                
            with col2:
                st.metric("Score", f"{ticker_data['score']:.2f}")
                st.metric("Probability Up", f"{ticker_data['prob_up']*100:.1f}%")
                st.metric("Holding Days", ticker_data['holding_days'])

if __name__ == '__main__':
    main()