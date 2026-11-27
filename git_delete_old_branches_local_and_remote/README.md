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
Run with `--tui` to be asked for every option instead of passing flags. The script shows the resulting command and asks before running it. Prompts use [gum](https://github.com/charmbracelet/gum) when it is installed (`brew install gum`) and plain terminal prompts otherwise.

`./git_delete_old_branches_local_and_remote.sh --tui`

You can also pick this script from the main launcher: `./helpful_scripts.sh` in the repo root.

### Examples
DRY RUN mode

`./git_delete_old_branches_local_and_remote.sh -r PATH_TO_REPO -s "3 days ago" -d`

Delete mode

`./git_delete_old_branches_local_and_remote.sh -r PATH_TO_REPO -s "2 weeks ago"`
