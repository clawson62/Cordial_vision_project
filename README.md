# Table of Contents
1. [Experiment 1: Can ChatGPT Vision Predict The Best Ad ](#experiment1)

   Results: 89-97% alignment with customer identity. 

3. [Experiment 2: Can ChatGPT Vision Predict Checkout Behavior](#experiment2)

   Results: UI affected behavior by 7% and customer identity affected behavior by 91-98%.

5. [Experiment 3:  Mimic Customer Journey For Lead Generation ](#experiment3)

    Results: Automate a webscraper to sign up for 1000's of D2C online stores, collect data
        about what companies need, and generate highly qualified leads.

6. [Reference Code](#code)


##### This is all unique work created for Cordial and not shared with any other organization.

---


<a id="experiment1"></a>
# Experiment  1: Can ChatGPT Vision Predict The Best Ad 

## Inspiration
https://cordial.com/clients/adore_me/

From the blog post above, I read how Adore Me customized ads for specific body types and product filters. This made me think about using ChatGPT to predict consumer behavior when viewing ads and instructing ChatGPT to behave like certain consumer identity profiles.

### Setup
I created two ads below targeting different consumer identities. 

Consumer Identity 1: (Right on ad)
```json
{"size":"44D",
"body type":"plus size",
"favorite bra padding":"lightly lined",
"favorite bra style":"full coverage"}
```
Consumer Identity 2: (Left on ad)

```json
{"size":"32B",
"body type":"small",
"favorite bra padding":"unlined",
"favorite bra style":"longlined"}
```

![Adore Me Ad Comparison](https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/adore_me_comparison.jpg)

# Experiment

I used the newest ChatGPT 4 with vision to ask which ad was preferred based on the consumer identity I gave ChatGPT.

Example Prompt:

You are a woman who Adore Me is sending an email ad to. You have the following consumer identity characteristics 

```json
{"size":"32B",
"body type":"small",
"favorite bra padding":"unlined",
"favorite bra style":"longlined"}
```

Respond with either "Left" if you prefer the ad on the left or respond with "Right" if you prefer the ad on the right.

# Results

![Adore Me Results](https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/Adore%20Me%20Results%20Pie%20Charts.png)

   

---

   

<a id="experiment2"></a>
# Experiment 2: Can ChatGPT Vision Predict Checkout Behavior 

## Inspiration: 

https://cordial.com/clients/eddie-bauer/

From the blog post above, I read how Eddie Bauer wanted to emphasize their Adventure Rewards program through personalization. Below is work detailing how ChatGPT can be instructed to behave like certain consumer identity profiles and predict the best UI, newsletter, email ad, or other image design.

### Setup
I tested the difference between two checkout user interfaces (UI) and two consumer identities to see if ChatGPT would prefer joining the Adventure Rewards

One UI has a more highlighted Adventure Rewards option.

One consumer identity specified the purchase_drivers of the consumer were "saving money", "deals", and "rewards".

<div style="text-align: center;">
   <img src="https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/eddie_bauer_one_green_button.jpg" style="width:250px; display:inline;" />
   <img src="https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/eddie_bauer_two_green_buttons.jpg" style="width:250px; display:inline;" />
</div>


# Experiment

I used the newest ChatGPT 4 with vision to understand how a customer with a specific consumer identity would use a checkout UI.

Example Prompt:

You are a consumer on the checkout page of Eddie Bauer with the following consumer identity:
```json
{"age":"35",
"gender":"Male",
"purchase_drivers":["saving money", "deals", "rewards"],
"ad_response":["deals","10% off sales"]} 
```
Click what you would do next on the page by saying "Click *screen object*". Do not reply with multiple sentences or explain your reasoning. 

# Results


## Test 1: Does ChatGPT Recognize UI elements If Aligned With Customer Identity
The first test showed that ChatGPT recognized the Join Rewards button was more apparent, and it chose it more often. When targeting customers with identities more prone to signing up for "deal" related programs, making those elements stand out could benefit the program.

![EB plot 1](https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/Eddie%20Bauer%20Results%20First%20%20Pie%20Chart.png)



     
     
     
     

   

   

## Test 2: Does ChatGPT Recognize UI elements If NOT Aligned With Customer Identity

The second test showed that without prompting ChatGPT to be interested in "savings related" content, ChatGPT chose to proceed with the purchase every time.
 
### 100% of the time ChatGPT chose to "Click Buy Now" when not given a customer identity.


   

---

   

<a id="experiment3"></a>
# Experiment 3: Mimic Customer Journey For Lead Generation 

### Background
<u> What if you could pretend to be 1000's of consumers and know how every online D2C business operates its webpage pop-ups, email notifications, SMS messages, and other marketing content?</u> 

I have expertise in designing complex scraping software that could do this and notify Cordial when companies:

- Don't have abandoned cart emails set-up
- Don't use exit-intent pop-ups
- Don't have rewards programs in place

For example, the website below *clothesmentor.com* does not have account creation, a rewards program, or exit-intent pop-ups. For every website, my script would return a json object like the one below and Cordial sales team can contact each lead **knowing what they need**.

![https://clothesmentor.com/](https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/clothesmentor_webpage.png)

### Example Website Information Breakdown

```json
{
  "Website Features": {
    "Newsletter Exists": true,
    "Newsletter Pop-up": false,
    "Exit-intent Pop-up": false,
    "Account Creation": {
      "Name": false,
      "Email": false,
      "Phone": false
    }
  },
  "App Exists": true
  "Email Notifications": {
    "Abandoned Cart": {
      "First": "None"
    },
    "Account Creation Message": false
  }
}
```

  

---

<a id="code"></a>


## Adore Me Code


```python
import openai
import pandas as pd
import time
import traceback
import json
import numpy as np
from collections import defaultdict
from openai import OpenAI
key = ""# insert key

client = OpenAI(api_key=key)


# api request for openai ChatGPT 4 vision
def gpt_vision(prompt,url):

    messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": prompt},
        {
          "type": "image_url",
          "image_url": {
            "url": url,
          },
        },
      ],
    }
  ]

    response = client.chat.completions.create(
            model="gpt-4-vision-preview", # newest model
            messages=messages,
            temperature=1.5,
            max_tokens=100
        )
    response_message = response
    return response_message
    
    
# create consumer identities
consumer_1_identity = {"size":"44D",
                       "body type":"plus size",
                       "favorite bra padding":"lightly lined",
                       "favorite bra style":"full coverage"}

consumer_2_identity = {"size":"32B",
                       "body type":"small",
                       "favorite bra padding":"unlined",
                       "favorite bra style":"longlined"}
# base prompt
adore_me_prompt = open("compare_adore_me_prompt.txt").read()

# url to ad
compare_url = "https://github.com/clawson62/Cordial_vision_project/blob/main/adore_me_comparison.jpg?raw=true"

consumer_1_responses = []
consumer_2_responses = []

# run each 100 times
for i in range(100):
    while True:
        try:
            adore_me_prompt = adore_me_prompt.replace("*identity*",str(consumer_1_identity))
            resp = gpt_vision(adore_me_prompt,compare_url)
            consumer_1_responses.append(resp.choices[0].message.content)
            break
            
        except:
            time.sleep(2)

for i in range(100):
    while True:
        try:
            adore_me_prompt = adore_me_prompt.replace("*identity*",str(consumer_2_identity))
            resp = gpt_vision(adore_me_prompt,compare_url)
            consumer_2_responses.append(resp.choices[0].message.content)
            break
            
        except:
            time.sleep(2)

# create data for pie charts            
labels_1,counts_1 = np.unique([i if i in ["Left","Right"] else "Other" for i in consumer_1_responses],return_counts=True)
labels_2,counts_2 = np.unique([i if i in ["Left","Right"] else "Other" for i in consumer_2_responses],return_counts=True)
```

## Eddie Bauer Code


```python
# url to ad
one_button_url = "https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/eddie_bauer_one_green_button.jpg"
two_buttons_url = "https://raw.githubusercontent.com/clawson62/Cordial_vision_project/main/eddie_bauer_two_green_buttons.jpg"

one_button_with_identity = []
two_buttons_with_identity = []
one_button_wo_identity = []
two_buttons_wo_identity = []

eddie_bauer_prompt = open("eddie_bauer_prompt.txt").read()

eddie_bauer_user_identity = {"age":"35",
                             "gender":"Male",
                             "purchase_drivers":["saving money", "deals", "rewards"],
                             "ad_response":["deals","10% off sales"]}
# run each 100 times
for i in range(100):
    while True:
        try:
            eddie_bauer_prompt = eddie_bauer_prompt.replace("*identity*",str(eddie_bauer_user_identity))
            resp = gpt_vision(eddie_bauer_prompt,one_button_url)
            one_button_with_identity.append(resp.choices[0].message.content)
            break
            
        except:
            time.sleep(2)
            
            
for i in range(100):
    while True:
        try:
            resp = gpt_vision(eddie_bauer_prompt,two_buttons_url)
            two_buttons_with_identity.append(resp.choices[0].message.content)
            break
            
        except:
            time.sleep(2)
            
            
            
            
eddie_bauer_prompt = open("eddie_bauer_prompt.txt").read()         
         
# change identity to simple
eddie_bauer_user_identity = {"age":"35",
                             "gender":"Male"}
for i in range(100):
    while True:
        try:
            eddie_bauer_prompt = eddie_bauer_prompt.replace("*identity*",str(eddie_bauer_user_identity))
            resp = gpt_vision(eddie_bauer_prompt,one_button_url)
            one_button_wo_identity.append(resp.choices[0].message.content)
            break
            
        except:
            time.sleep(2)
            
            
for i in range(100):
    while True:
        try:
            resp = gpt_vision(eddie_bauer_prompt,two_buttons_url)
            two_buttons_wo_identity.append(resp.choices[0].message.content)
            break
            
        except:
            time.sleep(2)

```
