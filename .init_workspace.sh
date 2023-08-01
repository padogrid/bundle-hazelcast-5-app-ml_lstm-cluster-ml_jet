#!/usr/bin/env bash

#
# .init_workspace.sh
#
# This script is invoked by 'install_bundle -init -checkout' and 'install_bundle -init -download'
# immediately after the bundle has been installed.
#
# The first argument is always the workspace name in which the bundle is installed.
#

# This script is executed only if 'jq' and 'open_vscode' executables are in the path.

WORKSPACE_NAME="$1"
WORKSPACE_PATH="$PADOGRID_WORKSPACES_HOME/$WORKSPACE_NAME"

#
# 1. Build the bundle
#
pushd $WORKSPACE_PATH/apps/ml_lstm/bin_sh > /dev/null
./build_app
popd > /dev/null

#
# 2. Create workspace.code-workspace only if the 'jq' and 'open_vscode' executables are available.
#    Add PYTHONPATH to workspace.code-workspace.
#
if [ "$(which jq)" == "" ]; then
   return
fi
if [ "$(which open_vscode)" == "" ]; then
   return
fi
WORKSPACE_CODE_WORKSPACE_FILE_NAME="workspace.code-workspace"
WORKSPACE_VSCODE_WORKSPACE_FILE="$WORKSPACE_PATH/$WORKSPACE_CODE_WORKSPACE_FILE_NAME"

# Initialize 'workspace.code-workspace' file
open_vscode -init -workspace "$WORKSPACE_NAME"

# Add PYTHONPATH in 'workspace.code-workspace'
cat $WORKSPACE_VSCODE_WORKSPACE_FILE  \
   | jq --argjson json '{"PYTHONPATH": "apps/ml_lstm/src/main/python"}' '. * {"settings": {"terminal.integrated.env.osx": ($json)}}' \
   | jq --argjson json '{"PYTHONPATH": "apps/ml_lstm/src/main/python"}' '. * {"settings": {"terminal.integrated.env.linux": ($json)}}' \
   > "/tmp/$WORKSPACE_CODE_WORKSPACE_FILE_NAME"
mv "/tmp/$WORKSPACE_CODE_WORKSPACE_FILE_NAME" "$WORKSPACE_VSCODE_WORKSPACE_FILE"

# Create .env with PYTHONPATH set
if [ -f "$WORKSPACE_PATH/.env" ]; then
   sed '/^PYTHONPATH/d' "$WORKSPACE_PATH/.env" > "$WORKSPACE_PATH/.env"
fi
echo "PYTHONPATH=$WORKSPACE_PATH/apps/ml_lstm/src/main/python" >> "$WORKSPACE_PATH/.env"
