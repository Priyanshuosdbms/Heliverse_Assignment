Continue the SystemRDL 2.0 conversion. This is batch <CURRENT_BATCH> of <TOTAL_BATCHES>. It contains registers <FIRST_REG_NAME> through <LAST_REG_NAME>.

Rules (same as before, repeated for emphasis):
- Emit only the new reg instantiations that belong inside the already-open addrmap <PERIPHERAL_NAME> block.
- Do NOT re-emit the addrmap header, enum definitions, or any register already converted in a previous batch.
- Every field must have: name, desc, sw, hw, and a reset value (= <value> on the instantiation line).
- Extract the reset value from the input key named reset, reset_value, resetValue, default, defaultValue, por, init, or value — whichever is present. If none is present, use = 0.
- Do not omit any register or any field. Do not add ellipsis or placeholders.
- Do not place comments inside reg or field bodies.
- After writing each field, verify the reset value in your output matches the input before continuing.
- If this is the final batch, close the addrmap block with };

Input (batch <CURRENT_BATCH> of <TOTAL_BATCHES>):

<PASTE TOON/JSON HERE>