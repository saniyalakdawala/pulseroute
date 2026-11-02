# Terraform local configuration

The configuration uses Terraform's built-in `terraform_data` resource and requires no cloud provider or credentials. `terraform init`, `validate`, and `plan` need no external providers. `apply` only records local runtime configuration in Terraform state; state must not be committed. This is not cloud infrastructure.
