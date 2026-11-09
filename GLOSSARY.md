# Refusal Direction & Refusal Distillation

Studies how chat models refuse harmful requests, and whether a larger model's refusal behaviour can be transferred to a smaller one.

## Language

### Models

**Teacher**:
The larger model whose refusal behaviour is the reference (here `gemma-3-4b-it`).
_Avoid_: big model, source model

**Student**:
The smaller model being compared against, and later trained toward, the Teacher (here `gemma-3-1b-it`).
_Avoid_: small model, target model

### Behaviour on harmful prompts

**Harmful prompt**:
An instruction asking for content a public-facing assistant should not produce.
_Avoid_: unsafe prompt, toxic prompt

**Refusal**:
A response in which the model declines the request, detected by the presence of a refusal phrase.
_Avoid_: rejection, block

**Harmful response**:
A response that actually contains harmful content, as judged by a safety classifier. A response can be neither a Refusal nor a Harmful response (e.g. vague or off-topic).
_Avoid_: jailbroken response, unsafe output

**Harmful rate**:
The fraction of Harmful prompts that produce a Harmful response. The primary measure of a model's safety.
_Avoid_: attack success rate (reserved for prompts with an attack applied)

**Disagreement**:
A Harmful prompt on which exactly one of Teacher and Student produces a Refusal.

**Safety advantage**:
The Teacher's Harmful rate is lower than the Student's by at least a set margin, and the paired difference is statistically significant.
