terraform {
  required_version = ">= 1.5"
}

resource "terraform_data" "runtime" {
  input = {
    service      = "pulseroute-api"
    port         = 8000
    metrics_path = "/metrics"
    model_path   = "../artifacts/champion.joblib"
  }
}

output "runtime_config" {
  value = terraform_data.runtime.output
}
