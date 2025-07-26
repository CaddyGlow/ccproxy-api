ENVIRONMENT="publish"
OWNER="caddyglow"
REPO="ccproxy-api"

# Get deployment IDs for the environment
deployment_ids=$(gh api repos/$OWNER/$REPO/deployments --jq ".[] | select(.environment==\"$ENVIRONMENT\") | .id")

# Deactivate and delete each deployment
for id in $deployment_ids; do
  echo "Processing deployment $id"
  gh api repos/$OWNER/$REPO/deployments/$id/statuses -f state=failed
  gh api repos/$OWNER/$REPO/deployments/$id -X DELETE
done
