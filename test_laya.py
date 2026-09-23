"""Smoke test for convaiinnovations/laya-multilingual.

Runs one predict() call: Hindi support email routed to a department,
plus a yes/no/unknown question. Verifies model loads and answers come
back with probabilities.
"""
import json

import laya

agent = laya.load("convaiinnovations/laya-multilingual")

result = agent.predict(
    {
        "body": (
            "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया है। "
            "कृपया अतिरिक्त भुगतान वापस कर दें।"
        )
    },
    {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle `body`?",
            "criteria": {
                "billing": "invoices, payments, refunds, duplicate charges",
                "technical": "bugs, errors, crashes, login problems",
                "sales": "pricing, demos, upgrades, new accounts",
            },
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the sender ask for money back?",
        },
    },
)

print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
