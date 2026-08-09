# Memory module context

Memory stores user/assistant conversation state and profile updates.

Persistence is downstream of the chat graph verifier. Blocked or failed responses must never be written as assistant answers. Background profile updates must remain bounded and must not change the response verification decision.
