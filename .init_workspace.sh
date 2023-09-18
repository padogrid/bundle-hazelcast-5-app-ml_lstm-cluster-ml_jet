#!/usr/bin/env bash

#
# .init_workspace.sh
#
# This script is invoked by 'install_bundle -init -checkout' and 'install_bundle -init -download'
# immediately after the bundle has been installed.
#
# The first argument is always the workspace name in which the bundle is installed.
#

WORKSPACE_NAME="$1"
WORKSPACE_PATH="$PADOGRID_WORKSPACES_HOME/$WORKSPACE_NAME"

if [ "$WORKSPACE_NAME" == "" ]; then
   echo >&2 "ERROR: The first argument must be a valid workspace name. Command aborted."
   exit 1
fi
if [ ! -d "$WORKSPACE_PATH" ]; then
   echo >&2 "ERROR: Specified workspace does not exist [$WORKSPACE_NAME]. Command aborted."
   exit 2
fi

# Switch workspace. This is required in order to build the bundle environment.
switch_workspace $WORKSPACE_NAME

#
# 1. Build the bundle
#
pushd $WORKSPACE_PATH/apps/ml_lstm/bin_sh > /dev/null
./build_app
popd > /dev/null

#
# 2. Create workspace.code-workspace only if the 'jq' and 'code' executables are available.
#    Add PYTHONPATH to workspace.code-workspace.
#
if [ "$(which jq)" == "" ]; then
   return
fi
if [ "$(which code)" == "" ]; then
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
