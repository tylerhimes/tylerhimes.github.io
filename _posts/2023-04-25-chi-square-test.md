---
layout: post
title: Assessing Campaign Performance Using Chi-Square Test For Independence
image: "/posts/ab-testing-title-img.png"
tags: [AB Testing, Hypothesis Testing, Chi-Square, Python]
---

In this project we apply **Chi-Square Test For Independence** (a Hypothesis Test) to assess the performance of two types of mailers that were sent out to promote a new service! 

# TABLE OF CONTENTS

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Concept Overview](#concept-overview)
- [02. Data Overview & Preparation](#data-overview)
- [03. Applying Chi-Square Test For Independence](#chi-square-application)
- [04. Analyzing The Results](#chi-square-results)
- [05. Discussion](#discussion)

___

# PROJECT OVERVIEW  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Earlier in the year, our client, a grocery retailer, ran a campaign to promote their new "Delivery Club" - an initiative that costs a customer $100 per year for membership, but offers free grocery deliveries rather than the normal cost of $10 per delivery.

For the campaign promoting the club, customers were put randomly into three groups: the first group received a low quality, low cost mailer; the second group received a high quality, high cost mailer; and the third group was a control group, receiving no mailer at all.

The client knows that customers who were contacted signed up for the Delivery Club at a far higher rate than the control group. Now the client wants to understand if there is a significant difference in sign-up rate between the cheap mailer and the expensive mailer. This will allow them to make more informed decisions in the future, with the overall aim of optimizing campaign ROI!

<br>
<br>
### Actions <a name="overview-actions"></a>

For this test, as it is focused on comparing the *rates* of two groups, we applied the **Chi-Square Test For Independence**. Full details of this test can be found in the dedicated section below.

**Note:** Another option when comparing "rates" is a test known as the **Z-Test For Proportions**. While we could certainly use this test here, we've chosen the **Chi-Square Test For Independence** because:
* The resulting test statistic for both tests will be the same.
* The **Chi-Square Test** can be represented using 2x2 tables of data - meaning it can be easier to explain to stakeholders.
* The **Chi-Square Test** can extend out to more than 2 groups - meaning the client can have one consistent approach to measuring signficance.

From the `campaign_data` table in the client database, we isolated customers that received "Mailer 1" (low cost) and "Mailer 2" (high cost) for this campaign, and excluded customers who were in the control group.

Our hypotheses and **Acceptance Criteria** for the test have been set as follows:
**Null Hypothesis:** There is no relationship between mailer type and signup rate. They are independent.
**Alternate Hypothesis:** There is a relationship between mailer type and signup rate. They are not independent.
**Acceptance Criteria:** 0.05

As a requirement of the **Chi-Square Test For Independence**, we aggregated this data down to a 2x2 matrix for `signup_flag` by `mailer_type` and fed this into the algorithm (using the `scipy` library) to calculate the Chi-Square Statistic, p-value, degrees of freedom, and expected values.

<br>
<br>

### Results & Discussion <a name="overview-results"></a>

Based upon our observed values, we can give this all some context with the sign-up rate of each group. We get:
* Mailer 1 (Low Cost): **32.8%** signup rate
* Mailer 2 (High Cost): **37.8%** signup rate

However, the **Chi-Square Test** gives us the following statistics:
* Chi-Square Statistic: **1.94**
* p-value: **0.16**
* Critical Value (for Acceptance Criteria of 0.05): **3.84**

Based on these statistics, we retain the **Null Hypothesis** and conclude that there is no relationship between mailer type and signup rate.

In other words - while we saw that the higher cost "Mailer 2" had a higher signup rate (37.8%) than the lower cost "Mailer 1" (32.8%), it appears that this difference is not significant at our **Acceptance Criteria** of 0.05.

Without running this **Hypothesis Test**, the client may have concluded they should always look to go with higher cost mailers. From what we've seen in this test, that may not be a great decision: it would result in them spending more, but not *necessarily* gaining any extra revenue as a result.

It's important to note that our results do not say that there *definitely isn't a difference between the two mailers* - we are only advising that we should not make any rigid conclusions *at this point*.  

Running more **A/B Tests** like this, gathering more data, and then re-running this test may provide us, and the client, more insight!

___

# CONCEPT OVERVIEW  <a name="concept-overview"></a>

<br>
#### A/B Testing

An **A/B Test** can be described as a randomized experiment containing two groups, A and B, that receive different experiences. Within an **A/B Test**, we look to understand and measure the response of each group and this information helps drive future business decisions.

Application of **A/B testing** can range from testing different online ad strategies, different email subject lines when contacting customers, or testing the effect of mailing customers a coupon vs a control group. Companies like Amazon are running these tests in an almost never-ending cycle - testing new website features on randomized groups of customers - all with the aim of finding what works best so they can stay ahead of their competition. Reportedly, Netflix will even test different images for the same movie or show to different segments of their customer base to see if certain images pull more viewers in.

<br>
#### Hypothesis Testing

A **Hypothesis Test** is used to assess the plausibility, or likelihood, of an assumed viewpoint based on sample data. In other words, it helps us assess whether a certain view we have about some data is likely to be true or not.

There are many different scenarios we can run **Hypothesis Tests** on, and they all have slightly different techniques and formulas. However, they all have some shared, fundamental steps and logic that underpin how they work.

<br>
**The Null Hypothesis**

In any **Hypothesis Test**, we start with the **Null Hypothesis**. The **Null Hypothesis** is where we state our initial viewpoint. In statistics - and specifically **Hypothesis Testing** - our initial viewpoint is always that the result is purely by chance or that there is no relationship or association between two outcomes or groups.

<br>
**The Alternate Hypothesis**

The aim of the **Hypothesis Test** is to look for evidence to support or reject the **Null Hypothesis**. If we reject the **Null Hypothesis**, that would mean we’d be supporting the **Alternate Hypothesis**. The **Alternate Hypothesis** is essentially the opposite viewpoint to the **Null Hypothesis** - that the result is *not* by chance, or that there *is* a relationship between two outcomes or groups.

<br>
**The Acceptance Criteria**

In a **Hypothesis Test**, before we collect any data or run any numbers, we specify an **Acceptance Criteria**. This is a **p-value** threshold at which we’ll decide to reject or support the **Null Hypothesis**. It is essentially a line we draw in the sand saying "*if I was to run this test many, many times, what proportion of those times would I want to see different results come out in order to feel comfortable or confident that my results are not just some unusual occurrence.*"

Conventionally, we set our **Acceptance Criteria** to 0.05, but this doesn't always have to be the case. If we need to be more confident that something did *not* occur through chance alone, we could lower this value down to something much smaller, meaning that we only come to the conclusion that the outcome was special or rare if it’s extremely rare.

So to summarize, in a **Hypothesis Test**, we test the **Null Hypothesis** using a **p-value** and then decide it’s fate based on the **Acceptance Criteria**.

<br>
**Types Of Hypothesis Test**

There are many different types of **Hypothesis Tests**, each of which is appropriate for use in differing scenarios depending on:
a) the type of data that you’re looking to test
b) the question that you’re asking of that data

In the case of our task here where we are looking to understand the difference in sign-up *rate* between two groups, we will utilize the **Chi-Square Test For Independence**.

<br>
#### Chi-Square Test For Independence

The **Chi-Square Test For Independence** is a type of **Hypothesis Test** that assumes observed frequencies for categorical variables will match the expected frequencies.

The *assumption* is the **Null Hypothesis**, which as discussed above is always the viewpoint that the two groups will be equal. With the **Chi-Square Test For Independence**, we look to calculate a statistic which, based on the specified **Acceptance Criteria**, will mean we either reject or support this initial assumption.

The *observed frequencies* are the true values that we’ve seen.

The *expected frequencies* are essentially what we would *expect* to see based on all of the data.

**Note:** Another option when comparing "rates" is a test known as the **Z-Test For Proportions**.  While we could certainly use this test here, we've chosen the **Chi-Square Test For Independence** because:
* The resulting test statistic for both tests will be the same.
* The **Chi-Square Test** can be represented using 2x2 tables of data - meaning it can be easier to explain to stakeholders.
* The **Chi-Square Test** can extend out to more than 2 groups - meaning the business can have one consistent approach to measuring signficance.

___

<br>
# DATA OVERVIEW & PREPARATION <a name="data-overview"></a>

In the client database, we have a `campaign_data` table that shows us which customers received each type of "Delivery Club" mailer, which customers were in the control group, and which customers joined the club as a result.

For this task, we are looking to find evidence that the Delivery Club sign-up rate for customers that received "Mailer 1" (low cost) was different to those who received "Mailer 2" (high cost) and thus from the `campaign_data` table we will just extract customers in those two groups, excluding customers who were in the control group.

In the code below, we:
* Load in the Python libraries required for importing the data and performing the chi-square test (using `scipy`)
* Import the required data from the `campaign_data` table
* Exclude customers in the control group, giving us a dataset with "Mailer 1" and "Mailer 2" customers only

<br>
```python
# Install the required Python libraries
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# Import campaign data
campaign_data = ...

# Remove customers who were in the control group
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]
```
<br>
A sample of this data (the first 10 rows) can be seen below:
<br>
<br>

| **customer_id** | **campaign_name** | **mailer_type** | **signup_flag** |
|---|---|---|---|
| 74 | delivery_club | Mailer1 | 1 |
| 524 | delivery_club | Mailer1 | 1 |
| 607 | delivery_club | Mailer2 | 1 |
| 343 | delivery_club | Mailer1 | 0 |
| 322 | delivery_club | Mailer2 | 1 |
| 115 | delivery_club | Mailer2 | 0 |
| 1 | delivery_club | Mailer2 | 1 |
| 120 | delivery_club | Mailer1 | 1 |
| 52 | delivery_club | Mailer1 | 1 |
| 405 | delivery_club | Mailer1 | 0 |
| 435 | delivery_club | Mailer2 | 0 |

<br>
In this DataFrame we have:
* customer_id
* campaign name
* mailer_type (either `Mailer1` or `Mailer2`)
* signup_flag (either `1` or `0`)

___

<br>
# APPLYING CHI-SQUARE TEST FOR INDEPENDENCE <a name="chi-square-application"></a>

<br>
#### State Hypotheses & Acceptance Criteria For Test

The very first thing we need to do in any form of **Hypothesis Test** is state our **Null Hypothesis**, our **Alternate Hypothesis**, and the **Acceptance Criteria** (more details on these in the section above).

In the code below, we code these explcitly and clearly so we can utilize them later to explain the results. We specify the common **Acceptance Criteria** value of 0.05.

<br>
```python
# Specify hypotheses & acceptance criteria for test
NULL_HYPOTHESIS = "There is no relationship between mailer type and signup rate.  They are independent."
ALTERNATE_HYPOTHESIS = "There is a relationship between mailer type and signup rate.  They are not independent."
ACCEPTANCE_CRITERIA = 0.05
```

<br>
#### Calculate Observed Frequencies & Expected Frequencies

As mentioned in the section above, in a **Chi-Square Test For Independence** the *observed frequencies* are the true values that we’ve seen - in other words the actual rates per group in the data itself. The *expected frequencies* are what we would *expect* to see based on *all* of the data combined.

The below code:
* Summarizes our dataset to a 2x2 matrix for `signup_flag` by `mailer_type`
* Based on this, calculates the:
    * Chi-Square Statistic
    * p-value
    * Degrees of Freedom
    * Expected Values
* Prints out the **Chi-Square Statistic** and **p-value** from the test
* Calculates the **Critical Value** based upon our **Acceptance Criteria** and the **Degrees Of Freedom**
* Prints out the **Critical Value**

<br>
```python
# Aggregate our data to get observed values
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values

# Run the chi-square test
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)

# Print chi-square statistic
print(chi2_statistic)
>> 1.94

# Print p-value
print(p_value)
>> 0.16

# Find the critical value for our test
critical_value = chi2.ppf(1 - ACCEPTANCE_CRITERIA, dof)

# Print critical value
print(critical_value)
>> 3.84
```
<br>
Based on our observed values, we can give this all some context with the sign-up rate of each group. We get:
* Mailer 1 (Low Cost): **32.8%** signup rate
* Mailer 2 (High Cost): **37.8%** signup rate

From this, we see that the higher cost mailer does lead to a higher signup rate. The results from our **Chi-Square Test** will provide us more information about how *confident* we can be that this difference is robust or if it might have occured by chance.

We have a **Chi-Square Statistic** of **1.94** and a **p-value** of **0.16**.  The critical value for our specified **Acceptance Criteria** of 0.05 is **3.84**.

**Note**: When applying the **Chi-Square Test** above, we use the parameter `correction = False` which means we are applying what is known as the *Yate's Correction*, which is applied when your **Degrees of Freedom** is equal to one. This correction helps to prevent overestimation of statistical signficance in this case.

___

<br>
# ANALYZING THE RESULTS <a name="chi-square-results"></a>

At this point, we have everything we need to understand the results of our **Chi-Square Test**. From the results above, we can see that since our resulting **p-value** of **0.16** is *greater* than our **Acceptance Criteria** of 0.05, we will *retain* the **Null Hypothesis** and conclude that there is no significant difference between the sign-up rates of "Mailer 1" and "Mailer 2".

We can make the same conclusion based upon our resulting **Chi-Square statistic** of **1.94** being *lower* than our **Critical Value** of **3.84**.

To make this script more dynamic, we can create code to automatically interpret the results and explain the outcome to us:

<br>
```python
# Print the results (based upon p-value)
if p_value <= ACCEPTANCE_CRITERIA:
    print(f"As our p-value of {p_value} is lower than our acceptance criteria of {ACCEPTANCE_CRITERIA} - we reject the null hypothesis and conclude that: {ALTERNATE_HYPOTHESIS}")
else:
    print(f"As our p-value of {p_value} is higher than our acceptance criteria of {ACCEPTANCE_CRITERIA} - we retain the null hypothesis and conclude that: {NULL_HYPOTHESIS}")

>> As our p-value of 0.16351 is higher than our acceptance criteria of 0.05 - we retain the null hypothesis and conclude that: There is no relationship between mailer type and signup rate.  They are independent.

# Print the results (based upon p-value)
if chi2_statistic >= critical_value:
    print(f"As our chi-square statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hypothesis and conclude that: {ALTERNATE_HYPOTHESIS}")
else:
    print(f"As our chi-square statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain the null hypothesis and conclude that: {NULL_HYPOTHESIS}")
    
>> As our chi-square statistic of 1.9414 is lower than our critical value of 3.841458820694124 - we retain the null hypothesis and conclude that: There is no relationship between mailer type and signup rate.  They are independent.
```
<br>
As we can see from the outputs of these print statements, we do indeed retain the **Null Hypothesis**. We could not find enough evidence that the sign-up rates for "Mailer 1" and "Mailer 2" were different - and thus conclude that there was no significant difference!

___

<br>
# DISCUSSION <a name="discussion"></a>

While we saw that the higher cost "Mailer 2" had a higher signup rate (37.8%) than the lower cost "Mailer 1" (32.8%), it appears that this difference is not significant, at least at our **Acceptance Criteria** of 0.05.

Without running this **Hypothesis Test**, the client may have concluded that they should always look to go with higher cost mailers. From what we've seen in this test, however, that may not be a great decision: it would result in them spending more, but not *necessarily* gaining any extra revenue as a result.

Our results here do not say that there *definitely isn't a difference between the two mailers* - we are only advising that we should not make any rigid conclusions *at this point*.  

Running more **A/B Tests** like this, gathering more data, and then re-running this test may provide us, and the client, more insight!
