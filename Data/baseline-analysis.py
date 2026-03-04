import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("Data/fraud_oracle.csv")


base_fraud_rate = df['FraudFound_P'].mean()
fraud_df = df[df['FraudFound_P'] == 1]


def fraudByCarPrice():
    sns.set_style("whitegrid")
    fraud_vehicle_price = df.groupby('VehiclePrice')['FraudFound_P'].mean().sort_values()

    plt.figure(figsize=(10,6))

    fraud_vehicle_price.plot(kind='bar')

    plt.axhline(base_fraud_rate, linestyle='--')
    plt.ylabel("Fraud Rate")
    plt.title("Fraud Rate by Vehicle Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def fraudByNumClaims():
    fraud_past_claims = df.groupby('PastNumberOfClaims')['FraudFound_P'].mean()

    plt.figure(figsize=(8,5))

    fraud_past_claims.plot(kind='bar')

    plt.axhline(base_fraud_rate, linestyle='--')
    plt.ylabel("Fraud Rate")
    plt.title("Fraud Rate by Past Number of Claims")
    plt.tight_layout()
    plt.show()


def fraudPieChart():
    fraud_counts = df['FraudFound_P'].value_counts()

    labels = ['Not Fraud', 'Fraud']

    plt.figure(figsize=(6,6))
    plt.pie(
        fraud_counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Fraud vs Non-Fraud Claims")
    plt.tight_layout()
    plt.show()



def witnessFraudChart():
    witness_counts = fraud_df['WitnessPresent'].value_counts()

    labels = witness_counts.index

    plt.figure(figsize=(6,6))
    plt.pie(
        witness_counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=["#d24949",'#66b3ff']
    )
    plt.title("Witness Presence in Fraudulent Claims")
    plt.tight_layout()
    plt.show()

def variableCounts():
    print(df['FraudFound_P'].value_counts())
    print('---')
    print(df['Sex'].value_counts())
    print('---')
    print(df['Fault'].value_counts())
    print('---')
    print(df['WitnessPresent'].value_counts())
    print('---')
    print(df['MaritalStatus'].value_counts())
    print('---')
