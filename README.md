# LCR
Liquidity Coverage Ratio (LCR) scenario analysis tool.




-------------------------------------------------------------------------------------------------------------
Referenced: https://stablebread.com/how-to-calculate-the-intrinsic-value-of-a-company-like-benjamin-graham/



Security Analysis
1. intrinsic value > market price -> IV - MP = MoS; more Mos = less risk
2. EPS, P/E, P/B, div yield -> value CS, PS, bonds

V = (EPS * (8.5 + 2g) * 4.4) / Y

1. EPS (TTM) = (NI - Pref Div) / Weighted Average Shares Outstanding --> replace to diluted EPS
2. adjust by the volatility of trailing EPS based on reports release (SEC)
3. diluted EPS (TTM) = (NI - Pref Div) / (WASO + unexercised employee stock options + convertible PS + convertible debt + warrants)

4. the "2" = growth multiplier; 1.0-1.5 (dig for background logic)
5. g -> analyst recommendation (dig behind logic)
6. 4.4 -> current 10-year average rate for AAA corporate bond (calculate trialing avg)
7. "Y" -> current yield on AAA corporate bonds -> wire API for up-to-date yield rate



---------------------------------------------------------------------------------------------------------
Step #1: Find/Calculate EPS
https://docs.google.com/spreadsheets/d/1A51A-6U8jWnppkbWJiz4tZkL5W9hShMaUNJdleh0HkE/edit?usp=sharing'


The EPS for Salesforce can be calculated using income statement data from the company's most-recent 10-K annual statement (for FY 2022 in this case):

EPS (TTM) = ($1.444B - $0) / 0.974B --> $1.48

In our case, we're using the TTM EPS calculation, so this results in an TTM EPS of $1.48.

Step #2: Estimate the EPS Growth Rate
The growth rate "g" is the rate at which EPS will grow over time (particularly 7-10 years). Again, this can be determined based on your understanding of the business, past historical growth rates, or by relying on analyst estimates for growth. Yahoo Finance is one popular financial data website where analyst estimates for growth are provided over the next 5 years, as seen below for CRM:

| Stablebread
Yahoo Finance: CRM Growth Estimates
As you can see, analysts estimate the EPS growth rate over the next 5 years (per annum) to be 15.13%. To keep things simple, we can use this growth rate assumption in Graham's CRM valuation model.

Step #3: Find the AAA Corporate Bond Yield
As previously discussed, the AAA corporate bond yield represents the discount rate for Graham's valuation method and changes periodically. Again, you can easily locate this data by visiting the FRED website. As of writing, Moody's AAA corporate bond yield is 3.76%, so we'll use this as our discount rate "Y."

Step #4: Solve For Graham's Formula
Now that we have all of the required inputs for Graham's valuation formula, all that's left is to solve for intrinsic value. To begin, Graham's revised valuation method to find the intrinsic value of CRM is shown below:

VCRM (revised) = ($1.48 * (8.5 + 2 * 15.13% * 100)) * 4.4) / 3.76 --> $67.13

Comparing Graham's intrinsic value calculation to the current stock price of CRM (currently ~$167) indicates that the stock is not worth buying at its current price, given that the intrinsic value of the stock is less than the current stock price (in this case, about 248% less (($167/$67) - 1)). This also does not take into consideration any margin of safety (as later discussed), so the stock is clearly overvalued from Graham's perspective.

Now, let's solve for Graham's formula again, but this time we'll adjust a few inputs based on the market and our familiarity with the business. Note that the viability and reasoning for adjusting these inputs were discussed earlier in this article.

For our example, I'll change the three following inputs:

P/E Base No-Growth Company: I calculated the cost of equity (re) for CRM to be 10.64% using the CAPM (re = 2.88% + 1.09*(10% - 2.88%)). This input now becomes 9.4 (1 / 10.64%) instead of 8.5.
Growth Rate: Instead of using an earnings growth rate from analysts off Yahoo Finance, I calculated the compound annual growth rate for CRM over the 5-year period (2018 - 2022). This growth rate is now 24.74% (($1.48 EPS / $0.49 EPS)(1/5) -1).
Growth Multiplier: The "2g" is already aggressive, and we're using a strong EPS growth rate of 24.74%, so I brought this down to "1g," which is what most value investors use anyways.
Using these three new inputs, Graham's intrinsic value calculation for CRM is shown below:

VCRM (adjusted) = ($1.48 * (9.4 + 1 * 24.74% * 100)) * 4.4%) / 3.76% --> $59.13

Despite making these changes to Graham's valuation inputs, CRM's stock would still appear to be significantly overvalued at its current stock price, even without applying a margin of safety.

Margin of Safety
"Margin of safety" is defined as the difference between the intrinsic value per share and the current market price of a company. This is a popular investing principle taught by Graham, which states that a security is only worth buying once its market price is substantially less than its estimated intrinsic value price.

Therefore, a margin of safety must be applied to Graham's stock valuation method, given that the intrinsic value per share calculation is not precise (as it's formula-driven and requires assumptions to be calculated). Even if your inputs and assumptions are reasonable, a margin of safety should always be applied. Otherwise, you may end up purchasing a business at an overvalued price, which is never ideal.

The margin of safety percentage you decide on simply depends on your confidence in the valuation, but should generally be on the higher end when lower discount rates are being used (as is the case for AAA corporate bond rates). For Graham's valuation formula, I would never use a margin of safety below 20-30%.

You can apply the margin of safety percentage to the intrinsic value per share of a company to calculate the appropriate buy price using the formula below:

Buy price (BP) = Intrinsic value per share * (1 - Margin of safety %)

As this formula shows, the greater the margin of safety, the more overvalued any particular stock will appear, and the more difficult it will be to find companies trading at your estimated buy price range, and vice versa.

Finally, after the margin of safety is applied to your intrinsic value calculation(s), you can then determine whether the stock is worth purchasing at its current stock price by comparing it to the buy price:

Buy price > Current market price: Consider buying the stock, as the current market price appears to be undervalued.
Buy price < Current market price: Consider selling or not buying the stock, as the current market price appears to be overvalued.
Keep in mind that this buy/sell recommendation is purely based on Graham's stock valuation formula and the current market price, and ignores all other fundamental, news, and market factors investors should examine as well before making an investment decision.

Salesforce Margin of Safety Example
To assess how a 30% margin of safety would affect CRM's intrinsic value per share calculations, see the two completed examples below for Graham's revised stock valuation method, and the adjusted Graham's stock valuation method I created:

BPCRM (revised) = $67.13 * (1 - 30%) --> $46.99

BPCRM (adjusted) = $59.13 * (1 - 30%) --> $41.39

Note that "BP" equals the stock's per-share buy price after applying the 30% margin of safety.

DJIA Benjamin Graham's Stock Valuation Method Analysis
To further analyze Benjamin Graham's stock valuation method for calculating the intrinsic value of a company, I applied the valuation model to all 30 companies in the U.S. Dow Jones Industrial Average (DJIA) Index. This is one of the most popular price-weighted indices and represents 30 large U.S. companies that cover a variety of different industries and sectors (around 25 and 10 respectively). Therefore, examining how Benjamin Graham's revised stock valuation method applies to different areas of the market, albeit, all large-cap stocks, may provide further insight on the best use-cases and limitations of the stock valuation method.

This analysis was completed in May 2022, and all companies that were valued were done so using Graham's most-recent (revised) valuation formula (from The Intelligent Investor):

V = (EPS * (8.5 + 2g) * 4.4) / Y

A default 50% margin of safety was applied to each intrinsic value per share calculation to calculate the buy price (BP) as follows:

BP = V * (1 - 50%)

All other inputs for Graham's stock valuation were kept the same, but the inputs for "g" and "Y" for each company are described below:

Growth Rate (g): For each Dow 30 company, I calculated the compound annual TTM EPS growth rate over the 5-year period (typically 2017 - 2021). This was accomplished by using the following formula: (EPSfinal / EPSbegin)(1/t) -1).
AAA Corporate Bond Yield (Y): This is just equal to Moody's AAA corporate bond yield, which as of writing is 3.76%.
Finally, the table below shows my results for each company with its respective sector and industry, the current stock price, buy price, and buy/sell recommendations:
