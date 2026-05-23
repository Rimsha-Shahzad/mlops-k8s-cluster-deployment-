resource "null_resource" "create_cluster" {
  provisioner "local-exec" {
    command = "kind create cluster --name terraform-k8s"
  }
}

resource "kubernetes_namespace" "dev" {
  metadata {
    name = "dev"
  }

  depends_on = [null_resource.create_cluster]
}
