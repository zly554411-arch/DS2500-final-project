Dataset: 
Around 1.05 million transactions from approximately 907K unique customers. 
90.78% of customers have one transaction.
9.22% of customers have multiple recorded transactions.
                        count    mean     std  min   25%    50%    75%     max   variance
payment_plan_days   1048575.0   66.01  102.48  0.0  30.0   30.0   30.0   450.0   10501.51
plan_list_price     1048575.0  281.74  435.09  0.0  99.0  149.0  149.0  2000.0  189301.88
actual_amount_paid  1048575.0  281.26  435.32  0.0  99.0  149.0  149.0  2000.0  189504.43
is_auto_renew       1048575.0    0.79    0.41  0.0   1.0    1.0    1.0     1.0       0.17
is_cancel           1048575.0    0.02    0.16  0.0   0.0    0.0    0.0     1.0       0.02

Subscription types: 
Most transactions are monthly (30-day plans), while annual subscriptions are relatively uncommon.
subscription_type
Monthly    892608     85.13%
Other      149710      0.60%
Annual       6257     14.28%


Auto-renewal: 
Overall auto-renewal rate is 78.53%.
About 92.07% of monthly transactions have auto-renewal enabled, compared with 0% of annual transactions. Auto-renewal also appears strongly associated with payment method.
is_auto_renew
1    823454
0    225121

Cancellation: 
Overall cancellation rate is about 2.47%, and it's 2.90% for monthly subscriptions. 
Cancellation rates also vary across payment methods, ranging from 0% to 8.23% among the 15 most common methods.
is_cancel
0    1022690
1      25885
