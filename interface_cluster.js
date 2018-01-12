window.clusters = undefined;
window.image_checked = undefined;
window.maxDepth = undefined;
window.done = undefined;
window.onload = function() {
    window.clusters = JSON.parse($('[name="l"]').val());
    window.maxDepth = window.clusters[0].length - 2;
    window.image_checked = window.clusters.map(function() { return false; })
    window.done = window.clusters.map(function() { return false; })
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

window.currClusters = [0];
function imageNotChecked(i) {
    if (window.image_checked[i] || window.done[i]) { return false; }
    var eq = true;
    window.currClusters.forEach(function(v, j) {
        if (eq && window.clusters[i][j+1] != window.currClusters[j]) { 
            eq = false;
        }
    })
    return eq;
}
function getImagesNotCheckedInCurrentCluster() {
    var not_checked = [];
    console.log("window clusters")
    console.log(window.clusters)
    window.clusters.forEach(function(c, i) { 
        if (imageNotChecked(i)) { not_checked.push(i); }
    });
    return not_checked;
}

function nextCluster(checked) {
    var not_checked = getImagesNotCheckedInCurrentCluster();
    if (checked) {
        not_checked.forEach(function(i) {
            window.image_checked[i] = true;
        })
    }
    if (!checked) { //if there was only a single image in that cluster, then we can rule out that image from future clusters
        if (not_checked.length == 1) {
            window.done[not_checked[0]] = true;
        }
    }
    
    //maintains a mutable copy of the checked states and propagates after each user interaction
    window.image_checked.forEach(function(t, i) { $('#i' + i).prop('checked', t); })

    if (getImagesNotCheckedInCurrentCluster().length > 0) {
        console.log("pushing")
        console.log(window.currClusters)
        window.currClusters.push(0);        
    }
    while (getImagesNotCheckedInCurrentCluster().length == 0 || getImagesNotCheckedInCurrentCluster().toString() == not_checked.toString()) {
        console.log(getImagesNotCheckedInCurrentCluster())
        console.log("incrementing")
        console.log(window.currClusters)
        window.currClusters[window.currClusters.length - 1]++;
        if (window.currClusters[window.currClusters.length - 1] > 1) {
            window.currClusters.pop();
            if (window.currClusters.length == 0) {
                console.log("next page")
                // nextPage();
                return;
            }
        }
    }
    //display all images in current cluster
    for (var i = 0; i < window.clusters.length; i++) {
        $('#m' + i).css('visibility', (imageNotChecked(i)) ? 'visible' : 'hidden')
    }
}

// currDepth = 0;
// currCluster = 0;
// currKMeansCluster = 0;

// KMeansClusters = 50;

// function clusterNotChecked(i) { 
//     return currDepth < window.maxDepth && !window.image_checked[i] && !window.done[i] && window.clusters[i][currDepth + 2] == currCluster && window.clusters[i][1] == currKMeansCluster;
// }

// function getImagesNotCheckedInCurrentCluster() {
    
//     not_checked = [];
//     window.clusters.forEach(function(c, i) { 
//         if (clusterNotChecked(i)) { not_checked.push(i); }
//     });
//     return not_checked;
// }
// function nextCluster(checked) {

//     console.log("curr Depth");
//     console.log(currDepth);
//     console.log("curr cluster");
//     console.log(currCluster);
//     not_checked = getImagesNotCheckedInCurrentCluster();
//     if (checked) {
//         not_checked.forEach(function(i) {
//             window.image_checked[i] = true;
//         })
//     }
//     if (!checked) { //if there was only a single image in that cluster, then we can rule out that image from future clusters
//         if (not_checked.length == 1) {
//             window.done[not_checked[0]] = true;
//         }
//     }

//     //maintains a mutable copy of the checked states and propagates after each user interaction
//     window.image_checked.forEach(function(t, i) { $('#i' + i).prop('checked', t); })
    
//     while (true)
//     {
//         currCluster++;
//         if (currCluster >= Math.pow(2, currDepth)) {
//             currCluster = 0;
//             currDepth++;
//         }
//         not_checked = getImagesNotCheckedInCurrentCluster();
//         if (currDepth >= window.maxDepth) {
//             currDepth = 0;
//             currKMeansCluster++;
//             if (currKMeansCluster >= KMeansClusters) {
//                 nextPage();
//                 return;
//             }
//         }
//         if (currDepth >= window.maxDepth || not_checked.length != 0) { break; }
//     }

//     for (var i = 0; i < window.clusters.length; i++) {
//         $('#m' + i).css('visibility', (clusterNotChecked(i)) ? 'visible' : 'hidden')
//     }
// }