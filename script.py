import pandas as pd

def create_fraud_features(df):
    """
    Analisa um DataFrame de transações para identificar comportamentos suspeitos.
    """
    DEVIATION_QUANTILE = 0.95 
    CARD_TEST_SMALL_AMOUNT = 5.00
    CARD_TEST_INCREASE_FACTOR = 8

    df_features = df.copy()
    df_features['transaction_date'] = pd.to_datetime(df_features['transaction_date'])
    df_features = df_features.sort_values(by=['card_number', 'transaction_date'])

    df_features['time_diff_minutes'] = df_features.groupby('card_number')['transaction_date'].diff().dt.total_seconds().div(60)
    df_features['is_high_velocity'] = df_features['time_diff_minutes'] <= 3
    df_features['distinct_cards_per_device'] = df_features.groupby('device_id')['card_number'].transform('nunique')
    df_features['distinct_devices_per_card'] = df_features.groupby('card_number')['device_id'].transform('nunique')

    df_features['card_transaction_count'] = df_features.groupby('card_number')['transaction_id'].transform('count')
    df_features['merchant_transaction_count'] = df_features.groupby('merchant_id')['transaction_id'].transform('count')

    df_features['card_p95_amount'] = df_features.groupby('card_number')['transaction_amount'].transform(lambda x: x.quantile(DEVIATION_QUANTILE))
    df_features['merchant_p95_amount'] = df_features.groupby('merchant_id')['transaction_amount'].transform(lambda x: x.quantile(DEVIATION_QUANTILE))

    is_above_card_p95 = df_features['transaction_amount'] > df_features['card_p95_amount']
    df_features['is_high_deviation_from_avg'] = is_above_card_p95 & (df_features['card_transaction_count'] > 1)

    is_above_merchant_p95 = df_features['transaction_amount'] > df_features['merchant_p95_amount']
    df_features['is_high_deviation_from_merchant_avg'] = is_above_merchant_p95 & (df_features['merchant_transaction_count'] > 1)

    df_features['next_transaction_amount'] = df_features.groupby('card_number')['transaction_amount'].shift(-1)
    
    is_small_current = df_features['transaction_amount'] < CARD_TEST_SMALL_AMOUNT
    is_large_next = df_features['next_transaction_amount'] > (df_features['transaction_amount'] * CARD_TEST_INCREASE_FACTOR)
    df_features['is_card_test_attempt'] = is_small_current & is_large_next.fillna(False)

    cols_to_drop = [
        'next_transaction_amount', 'card_transaction_count', 'merchant_transaction_count',
        'card_p95_amount', 'merchant_p95_amount'
    ]
    df_features = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])

    return df_features

df = pd.read_csv('./transactional-sample.csv')
enriched_df = create_fraud_features(df)

print("--- Resultados ---")
count_deviation = enriched_df['is_high_deviation_from_avg'].sum()
print(f"Número de transações com alto desvio (Percentil 95): {count_deviation}")

count_merchant_deviation = enriched_df['is_high_deviation_from_merchant_avg'].sum()
print(f"Número de transações com alto desvio do comerciante (Percentil 95): {count_merchant_deviation}")

count = enriched_df['is_card_test_attempt'].sum()
print(f"Número de tentativas de teste de cartão detectadas: {count}")

enriched_df.to_csv('enriched_df.csv', index=False)
