$(document).ready(function()
{
    var showing = 0;

    if (!mturk_isassigned())
    {
        mturk_acceptfirst();
    }
    else
    {
        mturk_showstatistics();
    }

    var parameters = mturk_parameters();
    if (!parameters["id"]) 
    {
        $("body").html("Missing ID query string.");
        return;
    }

    var marks = {};

    server_request("getjob", [parameters["id"]], function(data) {
        $(".category").html(data["category"]);

        var urls = [];
        for (var i = 0; i < data["windows"].length; i++)
        {
            urls.push("/images/" + data["windows"][i][1]);
        }
        preload(urls, function(p) { $("#debug").html(p); } );

        var lastimts = 0;

        $("#nextim").click(function() {
            if ((new Date()).getTime() - lastimts < 300)
            {
                alert("You are going too fast. Please look at the image again.");
                return;
            }
            if (marks[showing] == undefined)
            {
                alert("Please make a choice before proceeding.");
                return;
            }
            if (showing == data["windows"].length-1) return;
            $("#window" + showing).hide();
            showing++;
            update();

            lastimts = (new Date()).getTime();
        });
        $("#previm").click(function() {
            if (showing == 0) return;
            $("#window" + showing).hide();
            showing--;
            update();
        });

        $("#doesnotcontain").click(function() {
            marks[showing] = -1;
        });
        $("#doescontain").click(function() {
            marks[showing] = 1;
        });

        $("#submit").click(function() {
            var counter = 0;
            var payload = "[";
            for (var i in marks) {
                payload += "[" + data["windows"][i][0] + "," + marks[i] + "],";
                counter++;
            }
            payload = payload.substr(0, payload.length - 1) + "]";

            if (counter != data["windows"].length)
            {
                alert("You must have made a decision for every image before you can submit.");
                return;
            }

            mturk_submit(function(redirect) {
                server_post("savejob", [parameters["id"]], payload, function(data) {
                    redirect();
                });
            });
        });

        $(window).keypress(function(e) {
            if (e.which == 121) {
                $("#doescontain").click();
                $("#nextim").click();
            }
            if (e.which == 110) {
                $("#doesnotcontain").click();
                $("#nextim").click();
            }
        });

        function showwindow(i)
        {
            $("#windows").html("<img src='/images/" + data["windows"][i][1] + "'>");
        }

        function update()
        {
            showwindow(showing);
            $("#status").html("Showing image " + (showing+1) + " of " + data["windows"].length);

            if (marks[showing] == 1)
            {
                $("#doescontain").attr("checked", true);
            }
            else if (marks[showing] == -1)
            {
                $("#doesnotcontain").attr("checked", true);
            }
            else
            {
                $("#doesnotcontain").attr("checked", false);
                $("#doescontain").attr("checked", false);
            }
        }

        update();
    });
});
