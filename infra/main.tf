# Providers
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

# Variables
# Azure
variable "subscription_id" {
  type        = string
  description = "Azure subscription ID"
}


# Provider configurations
# Azure
provider "azurerm" {
  features {}

  subscription_id = var.subscription_id

  resource_provider_registrations = "none"
}

# Existing Azure resources
data "azurerm_subscription" "current" {}

data "azurerm_resource_group" "energy_prediction" {
  name = "Energy_Prediction"
}

# Outputs
output "subscription_name" {
  value = data.azurerm_subscription.current.display_name
}

output "tenant_id" {
  value = data.azurerm_subscription.current.tenant_id
}

output "resource_group_id" {
  value = data.azurerm_resource_group.energy_prediction.id
}

output "resource_group_location" {
  value = data.azurerm_resource_group.energy_prediction.location
}
