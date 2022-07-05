const offset = 20;
var is_dragging = false;
var show_legend = true, show_training = false, color_training = false;
const component_labels = {}, group_meta = {};
var zoom, selectedGroup = "class";
var nclass = 0, node_size_scale = 1, node_dist_scale = 1;
var name_to_id = {}, sort_block = false;
var node, link, hull;

var curve = d3.line()
    .curve(d3.curveCardinal.tension(0.85));
function drawCluster(d) {
    return curve(d.path);
}      

class MySet extends Set{
    add(elem){
      return super.add(typeof elem === 'object' ? JSON.stringify(elem) : elem);
    }
    has(elem){
      return super.has(typeof elem === 'object' ? JSON.stringify(elem) : elem);
    }
}

function create_hulls(groups,expands,offset) {
    // create point sets
    var hulls = [];
    for (var i = 0; i < Object.keys(groups).length; i++) {
        var l = [];
        if (expands[i] == false){
            continue;
        }
        for (var k = 0; k < groups[i].length; k++) {
            var n = groups[i][k];
            l.push([n.x-offset, n.y-offset]);
            l.push([n.x-offset, n.y+offset]);
            l.push([n.x+offset, n.y-offset]);
            l.push([n.x+offset, n.y+offset]);
        }   
        hulls.push({"group":[i], "path":d3.polygonHull(l)});
    }
  
    // create convex hulls
    return hulls;
  }

function update_network(data, init_nodes, graph, expand, groups, is_initial, dblclick_id, component_init_pos, selected_components) {
    var nodes = [], links = [], cnt = 0, node_map = {}, prev_locs = {};
    const n = data.nodes.length;
    let link_set = new MySet();
    if (is_initial == false) {
        graph.nodes.forEach(function(d) {
            prev_locs[d.id] = [d.x,d.y];
        });
    }
    // console.log(prev_locs);
    for (var k in expand){
        var i = parseInt(k);
        if (selected_components[groups[i][0].cid] == false) continue;
        if (expand[i]) {
            for (var j=0; j<groups[i].length; j++){
                var node = groups[i][j];
                node.size = 1;
                node.node_type = "node";
                // console.log(node);
                // if newly expanded, set initial position to be previous cluster centroid
                if (dblclick_id == n+i) {
                    node.x = prev_locs[dblclick_id][0];
                    node.y = prev_locs[dblclick_id][1];
                }
                // console.log(node);
                if (!(node.id in node_map)){
                    if (color_training & node.known_label){
                        node["pieChart"] = [{"color":node.label/nclass,"percent":100}];
                    }else{
                        node["pieChart"] = [{"color":node.prediction/nclass,"percent":100}];
                    }
                    nodes.push(node);
                    node_map[node.id] = cnt;
                    cnt += 1;
                }
            }
            // console.log(nodes);
            // console.log(groups[i]);
        }else{
            var node = {"id":n+i,"group":[i],"pieChart":group_meta[i],"size":groups[i].length,"node_type":"component","cid":groups[i][0].cid};
            if (node.id in prev_locs) {
                node.x = prev_locs[node.id][0];
                node.y = prev_locs[node.id][1];
            }else{
                node.x = component_init_pos[groups[i][0].cid][0];
                node.y = component_init_pos[groups[i][0].cid][1];
            }
            // if collapsed a cluster, set initial position to be mean x/y
            // console.log(node.group);
            // console.log(prev_locs);
            // console.log(groups[node.group[0]]);
            if (n+i == dblclick_id) {
                var all_x = 0, all_y = 0;
                groups[node.group[0]].forEach(function(d){
                    all_x += prev_locs[d.id][0];
                    all_y += prev_locs[d.id][1];
                });
                node.x = all_x/groups[i].length;
                node.y = all_y/groups[i].length;
            }
            nodes.push(node);
            node_map[n+i] = cnt;
            cnt += 1;
        }
    }
    for (var i=0; i < data.links.length; i++){
        var gsi = init_nodes[data.links[i].source].group;
        if (selected_components[groups[gsi[0]][0].cid] == false) continue;
        var gsj = init_nodes[data.links[i].target].group;
        var ni = init_nodes[data.links[i].source].id;
        var nj = init_nodes[data.links[i].target].id;
        gsi.forEach(function(gi){
            gsj.forEach(function(gj){
                var ei = gi+n;
                var ej = gj+n;
                if (!expand[gi] & !expand[gj]){
                   if (link_set.has([ej,ei]) == false){
                       link_set.add([ei,ej]);
                   } 
                }
                if (expand[gi] & expand[gj]){
                    if (link_set.has([nj,ni]) == false){
                        link_set.add([ni,nj]);
                    } 
                }
                if (!expand[gi] & expand[gj]){
                    if (link_set.has([nj,ei]) == false){
                        link_set.add([ei,nj]);
                    } 
                }
                if (expand[gi] & !expand[gj]){
                    if (link_set.has([ej,ni]) == false){
                        link_set.add([ni,ej]);
                    } 
                }
            });
        });
    }
    for (var i=0; i < init_nodes.length; i++){
        var gsi = init_nodes[i].group;
        if (selected_components[groups[gsi[0]][0].cid] == false) continue;
        var ni = init_nodes[i].id;
        gsi.forEach(function(gi){
            gsi.forEach(function(gj){
                if (gi != gj) {
                    var ei = gi+n;
                    var ej = gj+n;
                    if (!expand[gi] & !expand[gj]){
                    if (link_set.has([ej,ei]) == false){
                        link_set.add([ei,ej]);
                    } 
                    }
                    if (!expand[gi] & expand[gj]){
                        if (link_set.has([ni,ei]) == false){
                            link_set.add([ei,ni]);
                        } 
                    }
                    if (expand[gi] & !expand[gj]){
                        if (link_set.has([ej,ni]) == false){
                            link_set.add([ni,ej]);
                        } 
                    }
                }
            });
        });
    }
    // console.log(nodes);
    // console.log(node_map);
    link_set.forEach(function (d){
        d = JSON.parse(d);
        links.push({"source":d[0],"target":d[1]});
    });
    // console.log(data.nodes);
    // console.log(links);
    // console.log(nodes);
    // console.log(hulls);
    return {"nodes":nodes, "links":links, "hulls":create_hulls(groups,expand,offset)};
}

function draw_graph(data, init_nodes, graph, expand, groups, is_initial, dblclick_id, component_init_pos, selected_components,gDraw,parentWidth,parentHeight){
    graph = update_network(data, init_nodes, graph, expand, groups, is_initial, dblclick_id, component_init_pos, selected_components);
    is_initial = false;

    gDraw.selectAll("*").remove();

    // gDraw.selectAll("node").exit().remove();
    // gDraw.selectAll("hull").exit().remove();
    // gDraw.selectAll("link").exit().remove();

    // var width = +svg.attr("width"),
    //     height = +svg.attr("height");

    // var color = d3.scaleOrdinal(d3.schemeCategory10);

    if (! ("links" in graph)) {
        console.log("Graph is missing links");
        return;
    }

    var nodes = {};
    var i;
    for (i = 0; i < graph.nodes.length; i++) {
        nodes[graph.nodes[i].id] = graph.nodes[i];
        graph.nodes[i].weight = 1.01;
    }

    // the brush needs to go before the nodes so that it doesn't
    // get called when the mouse is over a node
    var gBrushHolder = gDraw.append('g');
    var gBrush = null;

    var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink()
                .id(function(d) { return d.id; })
                .distance(function(d) { 
                    if (d.source.node_type=='node' & d.target.node_type=='node'){
                        return 0;
                    }else{
                        return Math.min(Math.max(Math.min(d.source.size,d.target.size),50),200);
                    }
                })
            )
        .force("charge", d3.forceManyBody().strength(-200))
        .velocityDecay(.5)
        // .force("center", d3.forceCenter(parentWidth / 2, parentHeight / 2))
        .force("x", d3.forceX().x(function(d){return component_init_pos[d.cid][0]}))
        .force("y", d3.forceY().y(function(d){return component_init_pos[d.cid][1]}));
    
    gDraw.selectAll("path").remove();
    hull = gDraw.append("g")
        .attr("class", "hull")
        .selectAll("path")
        .data(graph.hulls)
        .enter().append("path")
        .attr("d", drawCluster)
        .style("fill", "lightblue")
        .attr("opacity", 0.3)
        .on("dblclick", function(d){
            // console.log(d);
            if (expand[d.group[0]]){
                expand[d.group[0]] = !expand[d.group[0]];
                simulation.stop();
                draw_graph(data, init_nodes, graph, expand, groups, is_initial, d.id, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
                if (selectedGroup != "class") update_color_scheme(selectedGroup);
                if (show_training != false) update_training_stroke(show_training);
            }
        });

    link = gDraw.append("g")
        .attr("class", "link")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

    node = gDraw.append("g")
        .attr("class", "node")
        .selectAll("g")
        .data(graph.nodes)
        .enter()
        .append("g")
        // .selectAll("circle")
        // .data(graph.nodes)
        // .enter().append("circle")
        // .attr("r", 5)
        // .attr("fill", function(d) { 
        //     if ('color' in d)
        //         return d.color;
        //     else
        //         return color(d.group[0]); 
        // })
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        // .on("click", function(d){
        //     console.log(d);
        // })
        .on("dblclick", function(d){
            tooltip.transition()
                .duration(50)
                .style("opacity", 0);
            // console.log(d);
            if (expand[d.group[0]] == false){
                expand[d.group[0]] = !expand[d.group[0]];
                simulation.stop();
                draw_graph(data, init_nodes, graph, expand, groups, is_initial, d.id, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
                if (selectedGroup != "class") update_color_scheme(selectedGroup);
                if (show_training != false) update_training_stroke(show_training);
            }
        })
        .on("mouseover", function(d, i){
            d3.select(this).transition()
                // .selectAll('circle').attr("stroke","black")
                .duration('50')
                .style("opacity", 0.7);
            tooltip.transition()
                .duration(50)
                .style("opacity", 1);
            // console.log(d3.event.pageX);
            // console.log(d3.event.pageY);
            var num;
            if (d.node_type == "node") {
                num = "Graph node id: "+d.id.toString();
                num += "<br>Prediction: "+class_names[d.prediction];
                num += "<br>Truth: "+class_names[d.label];
                num += "<br>Component id: "+d.cid.toString();
            }else{
                num = "Reeb node id: "+d.group[0].toString();
                num += "<br>Component id: "+d.cid.toString();
                num += "<br>Most common prediction: "+class_names[component_labels[d.cid]];
            }
            tooltip.html(num)
               .style("left", (d3.event.pageX + 10) + "px")
               .style("top", (d3.event.pageY - 15) + "px");
        })
        .on("mouseout", function(d, i){
            d3.select(this).transition()
                // .selectAll('circle').attr("stroke","white")
                .duration('50')
                .style("opacity", 1.0);
            tooltip.transition()
                .duration(50)
                .style("opacity", 0);
        });

    /* Draw the respective pie chart for each node */
    node.each(function (d) {
        NodePieBuilder.drawNodePie(d3.select(this), d.pieChart, {
            radius: Math.min(Math.pow(d.size,0.25)*10*node_size_scale,50),
            outerStrokeWidth: 3,
            parentNodeColor: "white",
            showPieChartBorder: false,
            showLabelText: false
        });
    });
      
    // add titles for mouseover blurbs
    node.append("title")
        .text(function(d) { 
            if ('name' in d)
                return d.name;
            else
                return d.id; 
        });

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);
    
    
    d3.select("#apply").on("click", function() {
        var text = document.getElementById("cids").value;
        for (var k in expand){
            expand[k] = false;
        }
        Object.keys(selected_components).forEach(function(k){
            selected_components[k] = false;
        });
        if (text == ""){
            Object.keys(selected_components).forEach(function(k){
                selected_components[k] = true;
            });
        }else{
            text.split(",").forEach(function(d){
                selected_components[parseInt(d)] = true;
            });
        }
        text = document.getElementById("selectButton").value;
        if (text != "all"){
            text = name_to_id[text];
            var selected_class = parseInt(text);
            Object.keys(selected_components).forEach(function(k){
                selected_components[k] = selected_components[k] & (component_labels[k] == selected_class);
            });
        }
        simulation.stop();
        draw_graph(data, init_nodes, graph, expand, groups, is_initial, null, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
        if (selectedGroup != "class") update_color_scheme(selectedGroup);
        if (show_training != false) update_training_stroke(show_training);
    });

    function ticked() {
        if (graph.hulls.length>0){
            hull.data(create_hulls(groups,expand,offset))
                .attr("d",drawCluster);
        }
        // update node and line positions at every step of 
        // the force simulation
        link.attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });
        
        d3.select("#drawing").selectAll("circle").attr("cx", function (d) {
                return d.x;
            })
            .attr("cy", function (d) {
                return d.y;
            });

        // node.attr("cx", function(d) { return d.x; })
        //     .attr("cy", function(d) { return d.y; });
    }

    d3.select('body').on('keydown', keydown);
    d3.select('body').on('keyup', keyup);

    var shiftKey = false;

    function keydown() {
        if (is_dragging == false) {
            shiftKey = d3.event.shiftKey;
        }
    }

    function keyup() {
        shiftKey = false;
    }

    function dragstarted(k) {
        is_dragging = true;
        // document.getElementById("demo").innerHTML = d.id;
        if (!d3.event.active) simulation.alphaTarget(0.9).restart();

        if(!shiftKey) {
            if (!k.selected && !shiftKey) {
                // if this node isn't selected, then we have to unselect every other node
                node.classed("selected", function(p) { return p.selected =  p.previouslySelected = false; });
            }
            d3.select(this).classed("selected", function(p) { k.previouslySelected = k.selected; return k.selected = true; });

            node.filter(function(d) { return d.selected; })
            .each(function(d) { //d.fixed |= 2; 
            d.fx = d.x;
            d.fy = d.y;
            //   d3.select(this).selectAll('circle').attr("stroke","black");
            });
            // d.dx = 0;
            // d.dy = 0;
        }else{
            node.filter(function(d) { return d.cid == k.cid; })
                .each(function(d) { //d.fixed |= 2; 
                d.fx = d.x;
                d.fy = d.y;
                //   d3.select(this).selectAll('circle').attr("stroke","black");
                });
            k.dx = 0;
            k.dy = 0;
        }
    }

    function dragged(k) {
        if(!shiftKey){
            //d.fx = d3.event.x;
        //d.fy = d3.event.y;
            node.filter(function(d) { return d.selected; })
            .each(function(d) { 
                d.fx += d3.event.dx;
                d.fy += d3.event.dy;
                // d.dx += d3.event.dx;
                // d.dy += d3.event.dy;
            })
        }else{
            node.filter(function(d) { return d.cid == k.cid;})
            .each(function(d) { 
                d.fx += d3.event.dx;
                d.fy += d3.event.dy;
            });
            k.dx += d3.event.dx;
            k.dy += d3.event.dy;
        }
    }

    function dragended(k) {
        is_dragging = false;
        if (!d3.event.active) simulation.alphaTarget(0);
        if(!shiftKey){
            k.fx = null;
            k.fy = null;
            node.filter(function(d) { return d.selected; })
            .each(function(d) { //d.fixed &= ~6; 
                d.fx = null;
                d.fy = null;
            })
        }else{
            k.fx = null;
            k.fy = null;
            node.filter(function(d) { return d.cid == k.cid; })
            .each(function(d) { //d.fixed &= ~6; 
                d.fx = null;
                d.fy = null;
            })
            component_init_pos[k.cid] = [
                component_init_pos[k.cid][0]+k.dx,
                component_init_pos[k.cid][1]+k.dy];
            // console.log(component_init_pos[k.cid]);
            simulation.force("x").initialize(graph.nodes);
            simulation.force("y").initialize(graph.nodes);
        }
    }
    d3.select("#select_color").on("change", function(d){
        selectedGroup = this.value;
        update_color_scheme(selectedGroup);
        if (show_training != false) update_training_stroke(show_training);
    });
    
    function update_color_scheme(v){
        // console.log("update");
        if (v=="class") {
            // console.log("class");
            node.each(function (d) {
                for (var p in d.pieChart){
                    d3.select(this)
                    .selectAll("circle#child-pie-"+p.toString())
                    .attr("opacity",1);
                    // .style("stroke-width",Math.min(Math.pow(d.size,0.25)*10,50));
                }
                d3.select(this)
                .selectAll("circle#parent-pie").attr("fill","white").attr("fill-opacity",1).attr("stroke","white");
            });
        }else{
            node.each(function (d) {
                for (var p in d.pieChart){
                    d3.select(this)
                    .selectAll("circle#child-pie-"+p.toString())
                    .attr("opacity",0);
                }
                var opacity_level = 0;
                if (d.node_type == "node") {
                    if (selectedGroup == "estimated_error") {
                        opacity_level += d.error_est;
                    }else{
                        opacity_level += (d.prediction != d.label);
                    }
                }else{
                    groups[d.group[0]].forEach(function(n){
                        if (selectedGroup == "estimated_error") {
                            opacity_level += n.error_est;
                        }else{
                            opacity_level += (n.prediction != n.label);
                        }
                    });
                    opacity_level /= groups[d.group[0]].length;
                }
                d3.select(this)
                .selectAll("circle#parent-pie").attr("fill","red").attr("fill-opacity",opacity_level).attr("stroke","transparent");
            });
        }
    }

    function update_training_stroke(show_training){
        if (show_training){
            node.each(function (d) {
                if (d.node_type == "node" & d.known_label){
                    d3.select(this)
                    .selectAll("circle#parent-pie").attr("stroke","black").attr("stroke-width",5);
                }
            });
        }else{
            node.each(function (d) {
                if (d.node_type == "node" & d.known_label){
                    if (selectedGroup == "class"){
                        d3.select(this)
                        .selectAll("circle#parent-pie").attr("stroke","white").attr("stroke-width",3);
                    }else{
                        d3.select(this)
                        .selectAll("circle#parent-pie").attr("stroke","transparent").attr("stroke-width",3);
                    }
                }
            });
        }
    }

    d3.select("#show_training").property("checked", show_training).on("change", function(d){
        show_training = !show_training; 
        update_training_stroke(show_training);
    });

    d3.select("#color_training").property("checked", color_training);
    d3.select("#color_training").on("change", function(d){
        color_training = !color_training; 
        const counter = {};
        Object.keys(groups).forEach(function(k){
            g = groups[k];
            counter[k] = {};
            for (var i=0; i<g.length; i++){
                if (g[i].known_label & color_training){
                    var l = g[i].label;
                }else{
                    var l = g[i].prediction;
                }
                if (l in counter[k]){
                    counter[k][l] += 1;
                }else{
                    counter[k][l] = 1;
                }
            }
            var cnt = g.length;
            var piechart = [];
            Object.keys(counter[k]).forEach(function(d){
                piechart.push({"color":parseInt(d)/nclass,"percent":100*counter[k][d]/cnt});
            });
            group_meta[k] = piechart;
        });
        simulation.stop();
        draw_graph(data, init_nodes, graph, expand, groups, is_initial, null, component_init_pos, selected_components, gDraw,parentWidth,parentHeight);
        if (selectedGroup != "class") update_color_scheme(selectedGroup);
        if (show_training != false) update_training_stroke(show_training);
    });
    
    d3.select("#smaller_node_size").on("click", function() {
        node_size_scale -= 0.05;
        node.each(function (d) {
            let radius = Math.min(Math.pow(d.size,0.25)*10*node_size_scale,50);
            let halfRadius = radius / 2;
            let halfCircumference = 2 * Math.PI * halfRadius;
            var percentToDraw = 0;
            for (var p in d.pieChart){
                percentToDraw += d.pieChart[p].percent;
                d3.select(this)
                .selectAll("circle#child-pie-"+p.toString())
                .attr("r", halfRadius)
                .style('stroke-width', radius)
                .style('stroke-dasharray',
                    halfCircumference * percentToDraw / 100
                    + ' '
                    + halfCircumference);
            }
            d3.select(this)
            .selectAll("circle#parent-pie").attr("r",radius);
        });
    });
    d3.select("#larger_node_size").on("click", function() {
        node_size_scale += 0.05;
        node.each(function (d) {
            let radius = Math.min(Math.pow(d.size,0.25)*10*node_size_scale,50);
            let halfRadius = radius / 2;
            let halfCircumference = 2 * Math.PI * halfRadius;
            var percentToDraw = 0;
            for (var p in d.pieChart){
                percentToDraw += d.pieChart[p].percent;
                d3.select(this)
                .selectAll("circle#child-pie-"+p.toString())
                .attr("r", halfRadius)
                .style('stroke-width', radius)
                .style('stroke-dasharray',
                    halfCircumference * percentToDraw / 100
                    + ' '
                    + halfCircumference);
            }
            d3.select(this)
            .selectAll("circle#parent-pie").attr("r",radius);
        });
    });
    d3.select("#output")
    .on('click', function(){
        var coords = [];
        node.each(function(d){
            if (d.node_type == "component") {
                coords.push({
                    "id": d.group[0],
                    "type": "Reeb net node",
                    "x": d.x,
                    "y": d.y
                });
            }else{
                coords.push({
                    "id": d.id,
                    "type": "original graph node",
                    "x": d.x,
                    "y": d.y
                });
            }
        });
        var blob = new Blob([JSON.stringify({"coords":coords})], {type: "text/plain;charset=utf-8"});  
        saveAs(blob, "coordinates.json");
    });

    return graph;
}

function createV4SelectableForceDirectedGraph(svg, graph, document) {
    // map each class id to its name
    for (var i in class_names){
        name_to_id[class_names[i]] = i;
    }
    // find the number of classes
    graph.nodes.forEach(function(d){nclass = Math.max(nclass,parseInt(d.label)+1);});
    let parentWidth = d3.select('#drawing').node().parentNode.clientWidth;
    let parentHeight = d3.select('#drawing').node().parentNode.clientHeight;
    const groups = {}, expand = {}, init_nodes = {}, counter = {}, component_init_pos = {}, selected_components = {}, components_by_label = {}, components_size={};
    // save a copy of the original data
    const data = JSON.parse(JSON.stringify(graph));
    // find out most common prediction in each component
    graph.nodes.forEach(function(d){
        d.group.forEach(function(g){
            if (g in groups) {
                groups[g].push(d);
            }else{
                groups[g] = [d];
            }
            expand[g] = false;
            init_nodes[d.id] = d;
        });
        component_init_pos[d.cid] = [0,0];
        if (d.cid in component_labels){
            components_size[d.cid] += 1;
            component_labels[d.cid][d.prediction] += 1;
        }else{
            components_size[d.cid] = 1;
            component_labels[d.cid] = new Array(nclass).fill(0);
            component_labels[d.cid][d.prediction] += 1;
        }
        selected_components[d.cid] = true;
    });
    for (k in component_labels){
        var l = component_labels[k].indexOf(Math.max(...component_labels[k]));
        if (l in components_by_label == false){
            components_by_label[l] = [];
        }
        component_labels[k] = l;
        components_by_label[l].push(parseInt(k));
    }
    var ncomponents = Object.keys(component_init_pos).length;
    var nrows = Math.round(Math.sqrt(ncomponents));
    nrows = Math.min(Math.round(ncomponents/nrows),nrows);
    // we assume each component can be fit into a square and 
    // we use packer to pack these squares as compact as possible
    // we first pack components within each class then combine components from different classes
    // after packing, we use the square center as the force center for each component
    // this can avoid components colliding
    function pack_components(sizes){
        var blocks = [], size_total = 0, ncomponents = sizes.length;
        sizes.forEach(function(k){
            var s = 15*Math.round(Math.sqrt(k[0]));
            size_total += s;
            blocks.push({"w":s,"h":s});
        });
        if (sort_block){
            blocks.sort(function(a,b) { return (b.h < a.h); });
        }
        var fit_cnt = 0, fit_scale = 0.5;
        var packer = new Packer(size_total*fit_scale, size_total*fit_scale);
        packer.fit(blocks);
        blocks.forEach(function(b){
            if (b.fit){
                fit_cnt += 1;
            }
        });
        // this means the packer size is too small to fit all squares, so we make it larger and retry
        while (fit_cnt < ncomponents){
            fit_cnt = 0;
            fit_scale += 0.1;
            packer = new Packer(size_total*fit_scale, size_total*fit_scale);
            packer.fit(blocks);
            blocks.forEach(function(b){
                if (b.fit){
                    fit_cnt += 1;
                }
            });
        }
        return [blocks, size_total*fit_scale];
    }

    function update_init_positions(blocks, sizes){
        if (sort_block){
            sizes.sort((a,b) => b[0]-a[0]);
        }
        var cnt = 0, block;
        for (var i = 0; i < sizes.length; i ++){
            while (true){
                block = blocks[cnt];
                cnt += 1;
                if (block.fit) break;
            }
            component_init_pos[sizes[i][1]] = [block.fit.y+block.w/2,block.fit.x+block.h/2];
        }
    }

    var packer_sizes = [];
    for (l in components_by_label){
        var sizes = [], tmp, blocks;
        components_by_label[l].forEach(function(k){
            sizes.push([components_size[k],k]);
        });
        tmp = pack_components(sizes);
        blocks = tmp[0];
        packer_sizes.push([5*tmp[1],l]);
        update_init_positions(blocks, sizes);
    }
    var tmp = pack_components(packer_sizes);
    var blocks = tmp[0];
    // if (sort_block){
    //     sizes.sort((a,b) => b[0]-a[0]);
    // }
    if (sort_block){
        packer_sizes.sort((a,b) => b[0]-a[0]);
    }
    var cnt = 0, block, center_loc = [0,0];
    for (var i = 0; i < packer_sizes.length; i ++){
        while (true){
            block = blocks[cnt];
            cnt += 1;
            if (block.fit) break;
        }
        // component_init_pos[sizes[i][1]] = [block.fit.y+block.w/2,block.fit.x+block.h/2];
        // center_loc[0] += component_init_pos[k][0];
        // center_loc[1] += component_init_pos[k][1];
        components_by_label[packer_sizes[i][1]].forEach(function(k){
            component_init_pos[k] = [component_init_pos[k][0]+block.fit.y,component_init_pos[k][1]+block.fit.x];
            center_loc[0] += component_init_pos[k][0];
            center_loc[1] += component_init_pos[k][1];
        })
    }
    // var fit_cnt = 0, fit_scale = 1;
    // var packer = new Packer(size_total*fit_scale, size_total*fit_scale);
    // packer.fit(blocks);
    // blocks.forEach(function(b){
    //     if (b.fit){
    //         fit_cnt += 1;
    //     }
    // });
    // // this means the packer size is too small to fit all squares, so we make it larger and retry
    // while (fit_cnt < ncomponents){
    //     fit_cnt = 0;
    //     fit_scale += 0.1;
    //     packer = new Packer(size_total*fit_scale, size_total*fit_scale);
    //     packer.fit(blocks);
    //     blocks.forEach(function(b){
    //         if (b.fit){
    //             fit_cnt += 1;
    //         }
    //     });
    // }
    // var cnt = 0, center_loc = [0,0];
    // for (l in components_by_label){
    //     components_by_label[l].forEach(function(k){
    //         var block = blocks[cnt];
    //         while (true){
    //             if (block.fit) break;
    //             cnt += 1;
    //             block = blocks[cnt];
    //         }
    //         component_init_pos[k] = [15*block.fit.y+15*block.w/2,15*block.fit.x+15*block.h/2];
    //         center_loc[0] += component_init_pos[k][0];
    //         center_loc[1] += component_init_pos[k][1];
    //         cnt += 1;
    //     });
    // }
    center_loc[0] /= ncomponents;
    center_loc[1] /= ncomponents;
    Object.keys(component_init_pos).forEach(function(k){
        var curr_loc = component_init_pos[k];
        component_init_pos[k] = [curr_loc[0]-center_loc[0]+parentWidth/2,curr_loc[1]-center_loc[1]+parentHeight/2];
    });
    // for each Reeb net node, compute its prediction proportion
    Object.keys(groups).forEach(function(k){
        g = groups[k];
        counter[k] = {};
        for (var i=0; i<g.length; i++){
            var l = g[i].prediction;
            if (l in counter[k]){
                counter[k][l] += 1;
            }else{
                counter[k][l] = 1;
            }
        }
        var cnt = g.length;
        var piechart = [];
        Object.keys(counter[k]).forEach(function(d){
            piechart.push({"color":parseInt(d)/nclass,"percent":100*counter[k][d]/cnt});
        });
        group_meta[k] = piechart;
    });
    // console.log(group_meta);
    const is_initial = true;

    var svg = d3.select('#drawing')
    .attr('width', parentWidth)
    .attr('height', parentHeight);

    // remove any previous graphs
    // svg.selectAll('.g-main').remove();

    var gMain = svg.append('g')
    .classed('g-main', true);

    var rect = gMain.append('rect')
    .attr('width', parentWidth)
    .attr('height', parentHeight)
    .style('fill', 'transparent');

    var gDraw = gMain.append('g');

    zoom = d3.zoom()
    .on('zoom', zoomed);

    gMain.call(zoom).on("dblclick.zoom", null);


    function zoomed() {
        gDraw.attr('transform', d3.event.transform);
    }
    
    d3.select("#zoom_in").on("click", function() {
        zoom.scaleBy(gMain.transition().duration(200), 1.2);
    });
    d3.select("#zoom_out").on("click", function() {
        zoom.scaleBy(gMain.transition().duration(200), 0.8);
    });
    
    function zoomFit(transitionDuration) {
        var bounds = gDraw.node().getBBox();
        var parent = gDraw.node().parentElement;
        var fullWidth  = parent.clientWidth  || parent.parentNode.clientWidth,
            fullHeight = parent.clientHeight || parent.parentNode.clientHeight;
        var width  = bounds.width,
            height = bounds.height;
        var midX = bounds.x + width / 2,
            midY = bounds.y + height / 2;
        if (width == 0 || height == 0) return; // nothing to fit
        var scale = 0.85 / Math.max(width / fullWidth, height / fullHeight);
        var translate = [
            fullWidth  / 2 - scale * midX,
            fullHeight / 2 - scale * midY
        ];
        
        var transform = d3.zoomIdentity
        .translate(translate[0], translate[1])
        .scale(scale);

        gMain
        .transition()
        .duration(transitionDuration || 0) // milliseconds
        .call(zoom.transform, transform);
    }

    function draw_legend(legend){
        for (var i=0; i<nclass; i++){
            legend.append("circle").attr("cx",50).attr("cy",50+45*i).attr("r", 20).style("fill", d3.interpolateRainbow(i/nclass));
            legend.append("text").attr("x", 350).attr("y", 50+45*i).text(class_names[i]).style("font-size", "15px").attr("alignment-baseline","right");
        }
    }

    d3.select("#fit").on("click", function() {
        zoomFit(1000);
    });
    d3.select("#selectButton").append("option").text("all");
    for (var i = 0; i < nclass; i++){
        d3.select("#selectButton").append("option").text(class_names[i]);
    }
    var allGroup = ["class","estimated_error","true_error"];
    d3.select("#select_color")
        .selectAll('myOptions')
        .data(allGroup)
        .enter()
        .append('option')
        .text(function (d) { return d; }) 
        .attr("value", function (d) { return d; });
    
    var legend = d3.select("#drawing_legend").style("border","0px solid black");
    draw_legend(legend);
    d3.select("#show_legend").property("checked", show_legend).on("change", function(d){
        show_legend = !show_legend; 
        if (show_legend){
            draw_legend(legend);
        }else{
            legend.selectAll("circle").remove();
            legend.selectAll("text").remove();
        }
    });
     
    var ret = draw_graph(data,init_nodes,graph,expand,groups,is_initial,null,component_init_pos,selected_components,gDraw,parentWidth,parentHeight);
    setTimeout(zoomFit, Math.min(Math.sqrt(Object.keys(groups).length)*40,3000), 1000);
    return ret;
};
