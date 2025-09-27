import pandas as pd

def create_fraud_features(df):
    """
    Analyzes a DataFrame to create features that help identify potential fraud.
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

def print_fraud_analysis_results(df):
    """
    Prints summary results of the fraud analysis.
    """
    print("--- Results ---")

    count_deviation = df['is_high_deviation_from_avg'].sum()
    print(f"Number of transactions with high deviation (95th Percentile): {count_deviation}")

    count_merchant_deviation = df['is_high_deviation_from_merchant_avg'].sum()
    print(f"Number of transactions with high merchant deviation (95th Percentile): {count_merchant_deviation}")

    count_card_test = df['is_card_test_attempt'].sum()
    print(f"Number of card test attempts detected: {count_card_test}")

def prepare_data_for_export(df):
    """
    Prepares data for export, converting columns to appropriate types.
    """
    df_export = df.copy()
    df_export['has_cbk'] = df_export['has_cbk'].astype(int)

    flag_columns = [
        'is_high_velocity', 
        'is_high_deviation_from_avg', 
        'is_high_deviation_from_merchant_avg', 
        'is_card_test_attempt'
    ]

    for col in flag_columns:
        if col in df_export.columns:
            df_export[col] = df_export[col].astype(int)

    return df_export

def export_enriched_data(df, filename='enriched_transaction_analysis.csv'):
    """
    Exports enriched data to a CSV file.
    """
    df.to_csv(filename, index=False)
    print(f"\nFile '{filename}' created successfully!")

def create_device_summary(df):
    """
    Creates a summary by device with chargeback metrics.
    """
    summary_by_device = df.groupby('device_id').agg(
        total_transactions=('transaction_id', 'count'),
        chargeback_count=('has_cbk', 'sum'),
        distinct_cards_used=('card_number', 'nunique')
    ).reset_index()

    summary_by_device['chargeback_rate'] = (
        summary_by_device['chargeback_count'] / summary_by_device['total_transactions']
    )

    summary_by_device = summary_by_device.sort_values(by='chargeback_count', ascending=False)

    return summary_by_device

def export_device_summary(summary_df, filename='summary_by_device.csv'):
    """
    Exports device summary to a CSV file.
    """
    summary_df.to_csv(filename, index=False)
    print(f"\nSummary file '{filename}' created successfully!")

def main():
    """
    Main function that executes the entire analysis pipeline.
    """
    df = pd.read_csv('./transactional-sample.csv')

    enriched_df = create_fraud_features(df)

    print_fraud_analysis_results(enriched_df)

    export_ready_df = prepare_data_for_export(enriched_df)
    export_enriched_data(export_ready_df)

    device_summary = create_device_summary(export_ready_df)
    export_device_summary(device_summary)

if __name__ == "__main__":
    main()
