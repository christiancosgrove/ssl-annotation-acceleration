document.onload = function() {
    clusters = JSON.parse($('[name="l"]').val())
    maxDepth = clusters[0].length
}
dnow = new Date()

function getParameterByName(name, url) {
    if (!url) url = window.location.href;
    name = name.replace(/[\[\]]/g, "\\$&");
    var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, " "));
}

function nextPage() {
    
    var aid=$('[name="assignmentId"]').val()
    var dat = {
        'c':$('[name="c"]').val(),
        's':$('[name="s"]').val(),
        'p':$('[name="p"]').val(),
        'assignmentId':aid,
        'workerId':$('[name="workerId"]').val(),
        'time': (new Date() - dnow),
        'clustering':$('[name="clustering"]').val(),
        'responses':[]
    }

    for (var i = 0; i < NUM_IMAGES; i++) {
        if ($('#i' + i).is(':checked')) {
            dat['responses'][i] = 1
        } else {
            dat['responses'][i] = -1
        }
    }
    // if (aid != 'ASSIGNMENT_ID_NOT_AVAILABLE') {
        if (!submitted)
        {
            submitted = true
            $.ajax({type:'POST', url:'/submit', data: JSON.stringify(dat), contentType:'application/json'}).done(function() {
                document.getElementById('form1').submit()
            })
        }
    // }
}

document.onkeypress = function(e) {
    e = e || window.event;
    if (e.keyCode == 110) { //yes
        nextCluster(false);
    } else if (e.keyCode == 109) { // no
        nextCluster(true);
    }
}

submitted = false;
currItemIndex = 0;
function nextItem(checked) {
    
    $('#i' + currItemIndex).parent().hide();
    $('#i' + currItemIndex).prop('checked', checked);
    currItemIndex++;
    if (currItemIndex < NUM_IMAGES) { 
        $('#i' + currItemIndex).parent().show();
    } else {
        nextPage();
    }
}


currDepth = 0;
currCluster = 0;
function getClustersNotChecked() {
    
    not_checked = [];
    clusters.forEach(function(c, i) { 
        if (currDepth < maxDepth && !$('#i' + i).is(':checked') && c[currDepth] == currCluster) { not_checked.push(i); }
    });
    return not_checked;
}
function nextCluster(checked) {

    not_checked = getClustersNotChecked();
    if (checked) {
        not_checked.forEach(function(i) {
            $('#i' + i).prop('checked', checked);
        })
    }
    while (currDepth < maxDepth && not_checked.length == 0)
    {
        not_checked = getClustersNotChecked();
        currCluster += 1;
        if (currCluster >= Math.pow(2, currDepth)) {
            currCluster = 0;
            currDepth += 1;
        }
    }
    if (currDepth >= maxDepth) {
        nextPage();
    }
}