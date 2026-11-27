# GIT delete old local and remote branches

This Bash script deletes all branches (local and remote) older than passed date.

### Requirement
`egrep`
`git`

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**d**|Dry run|false|
|**r**|Repository folder|-|
|**s**|Since date|`6 months ago`|
|**h**|Help|-|

### Interactive mode
Run with `--tui` to open the terminal UI of this repo on this script: edit the options in a form, see the resulting command and run it with a live log. The flags above keep working as before.

`./git_delete_old_branches_local_and_remote.sh --tui`

You can also pick this script from the full list with `./helpful_scripts_tui.py` in the repo root.

### Examples
DRY RUN mode

`./git_delete_old_branches_local_and_remote.sh -r PATH_TO_REPO -s "3 days ago" -d`

Delete mode

`./git_delete_old_branches_local_and_remote.sh -r PATH_TO_REPO -s "2 weeks ago"`
