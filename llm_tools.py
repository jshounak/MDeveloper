{
  "name": "predict_vagmd",
  "description": "Predict V-AGMD condenser outlet temperature and distillate flux.",
  "parameters": {
    "type": "object",
    "properties": {
      "T_mem_in": {"type": "number"},
      "T_con_in": {"type": "number"},
      "S": {"type": "number"},
      "v_chan": {"type": "number"},
      "vac": {"type": "number"},
      "L_type": {"type": "integer"},
      "sp_type": {"type": "integer"},
      "spa_type": {"type": "integer"},
      "membrane": {"type": "integer"}
    },
    "required": [
      "T_mem_in", "T_con_in", "S", "v_chan", "vac",
      "L_type", "sp_type", "spa_type", "membrane"
    ]
  }
}