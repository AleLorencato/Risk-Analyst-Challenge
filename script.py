import pandas as pd

def create_fraud_features(df):
    """
    Analisa um DataFrame de transações para identificar comportamentos suspeitos.
    """

    df_features = df.copy()
    df_features['transaction_date'] = pd.to_datetime(df_features['transaction_date'])
    df_features = df_features.sort_values(by=['card_number', 'transaction_date'])

    # 2. Feature de Velocidade (Velocity Checking)
    df_features['time_diff_minutes'] = df_features.groupby('card_number')['transaction_date'].diff().dt.total_seconds().div(60)

    df_features['is_high_velocity'] = df_features['time_diff_minutes'] <= 3

    # 3. Features de Dispositivo e Vínculos (Device Checking & Link Analysis)
    df_features['distinct_cards_per_device'] = df_features.groupby('device_id')['card_number'].transform('nunique')

    df_features['distinct_devices_per_card'] = df_features.groupby('card_number')['device_id'].transform('nunique')

    # 4. Feature de Teste de Cartão (Card Testing)
    df_features['next_transaction_amount'] = df_features.groupby('card_number')['transaction_amount'].shift(-1)

    small_amount_threshold = 5.00
    increase_factor = 10
    is_small_current = df_features['transaction_amount'] < small_amount_threshold
    is_large_next = df_features['next_transaction_amount'] > (df_features['transaction_amount'] * increase_factor)
    df_features['is_card_test_attempt'] = is_small_current & is_large_next

    count = df_features['is_card_test_attempt'].sum()
    print(f"Número de tentativas de teste de cartão detectadas: {count}")

    # Ideias do que implementar:
    ## Verificar se as features que criei realmente se correlacionam com a fraude (has_cbk)
    ## Desvio da Média do Usuário/Cartão
    ## Desvio da Média do Comerciante (merchant_id)
    ## Contagem de Transações em Janelas de Tempo (Rolling Windows)
    ## Anomalia Cronológica (Hora do Dia):
    ## Primeira Transação do Usuário/Cartão

    df_features = df_features.drop(columns=['next_transaction_amount'])

    return df_features

df = pd.read_csv('./transactional-sample.csv')

enriched_df = create_fraud_features(df)

enriched_df.to_csv('enriched_transaction_analysis.csv', index=False)

print("DataFrame com as novas features de fraude:")
print(enriched_df[['card_number', 'has_cbk', 'is_high_velocity', 'distinct_cards_per_device', 'is_card_test_attempt']].head(10))
