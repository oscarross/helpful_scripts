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

### Examples
DRY RUN mode

`./git_delete_old_branches_local_and_remote.sh -r PATH_TO_REPO -s "3 days ago" -d` 

Delete mode

`./git_delete_old_branches_local_and_remote.sh -r PATH_TO_REPO -s "2 weeks ago"`

