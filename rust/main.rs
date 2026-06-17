use core::ops::ControlFlow;
use meshx::io::save_trimesh_ascii;
use std::collections::HashSet;
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use duration_string::DurationString;

use indicatif::{ProgressBar, ProgressStyle};

use clap::Parser;

use lazy_static::lazy_static;

use meshx::TriMesh;

use nalgebra::distance;
use nalgebra::geometry::Point3;

use itertools::Itertools;

// Parse the arguments and store them in a global variable
//  They never change, so a global is safe to use
lazy_static! {
    static ref ARGS: Args = Args::parse();
}

// Define what arguments the user can input.
#[derive(Parser, Debug)]
struct Args {
    /// Number of rings. Must be greater than 0
    #[arg()]
    rings: NonZeroUsize,

    // The number of triangles per point. Must be 7 or greater
    #[arg(default_value_t = 7)]
    triangles_per_point: usize,

    /// Error margin for distance calculations
    #[arg(short, default_value_t = 10e-6)]
    error_margin: f64,

    // TODO: Better error reporting
    /// Time to run before exiting: [0-9]+(ns|us|ms|[smhdwy]). eg: 5m, 15s, 100m
    #[arg(short, default_value = "15s")]
    time: DurationString,

    /// Enable debug printing
    #[arg(short, action)]
    debug: bool,

    #[arg(long, action)]
    animate: bool,
}

impl Args {
    /// Convert the user given time string to a duration
    fn time(&self) -> Duration {
        self.time.into()
    }
}

fn truncate(value: f64) -> isize {
    value.floor() as isize
}

/// A Vector type
type Vec3 = Point3<f64>;

struct CollisionGroups {
    // Maybe should be [[[HashShet<usize>]]]
    groups: HashMap<(isize, isize, isize), HashSet<usize>>,
}

impl CollisionGroups {
    fn new(_rings: usize) -> Self {
        let groups = HashMap::new();
        Self {
            groups,
        }
    }
    
    fn get(&self, x: isize, y: isize, z: isize) -> &HashSet<usize> {
        &self.groups[&(x, y, z)]
    }
    
    fn get_mut(&mut self, x: isize, y: isize, z:isize) -> &mut HashSet<usize> {
        self.groups.entry((x, y, z)).or_default()
    }

    fn add_point(&mut self, point: usize, group: [isize; 3]) {
        let [gx, gy, gz] = group;
        self.get_mut(gx, gy, gz)
            .insert(point);
    }

    fn move_point(&mut self, point: usize, old_group: Vec3, new_group: Vec3) {
        if old_group != new_group {
            // let old_c = old_group.coords;
            let old_g = [
                truncate(old_group.x),
                truncate(old_group.y),
                truncate(old_group.z),
            ];
            // let new_c = *(new_group.coords);
            let new_g = [
                truncate(new_group.x),
                truncate(new_group.y),
                truncate(new_group.z),
            ];
            //println!("{old_c:?} {new_c:?}");
            let [oldx, oldy, oldz] = old_g;
            let [newx, newy, newz] = new_g;
            //println!("Moving point {point} from [{ox}, {oy}, {oz}] to [{nx}, {ny}, {nz}]");
            // let Some(oi) = self.groups[ox][oy][oz].iter().position(|x| *x == point) else {
            //     panic!("index {point} not found in {:?}", self.groups[ox][oy][oz]);
            // };
            self.get_mut(oldx, oldy, oldz).remove(&point);
            self.get_mut(newx, newy, newz).insert(point);
        }
    }

    fn get_neighbor_groups(&self, group: [isize; 3]) -> Vec<[isize; 3]> {
        let [gx, gy, gz] = group;
        let nums = [-2, -1, 0, 1, 2];
        nums.iter()
            .cartesian_product(nums)
            .cartesian_product(nums)
            .map(|((x, y), z)| [*x, y, z])
            .map(|[x, y, z]| [gx + x, gy + y, gz + z])
            .filter(|[x, y, z]| self.groups.contains_key(&(*x, *y, *z)))
            .collect()
    }

    fn get_neighbor_points(&self, point: usize, group: [isize; 3]) -> Vec<usize> {
        self
            .get_neighbor_groups(group)
            .into_iter()
            .flat_map(|[x, y, z]| self.get(x, y, z))
            .copied()
            .unique()
            .filter(|p| *p > point)
            .collect()
    }
}

/// The base struct
/// This contains all the points, and keeps track of which points form lines
/// and triangles. Points are shared between triangles, so if you move one point,
/// the point in all triangles that share that point move as well
struct Mesh {
    points: Vec<Vec3>,
    pairs: Vec<[usize; 2]>,
    //duals: Vec<[usize; 2]>,
    collision_groups: CollisionGroups,
    tris: Vec<[usize; 3]>,
}

impl Mesh {
    /// Creates a new mesh with the proper amount of points
    pub fn new(rings: NonZeroUsize) -> Mesh {
        // Initialize random for later
        let rng = fastrand::Rng::new();

        // Generate number of points per ring
        // Formula found https://oeis.org/A001354
        let mut ring_counts = vec![0, ARGS.triangles_per_point];
        for i in 2..=usize::from(rings) {
            ring_counts
                .push((ARGS.triangles_per_point - 4) * ring_counts[i - 1] - ring_counts[i - 2]);
        }
        //ring_counts[0] = 1;
        ring_counts
            .iter_mut()
            .filter(|x| **x == 0)
            .for_each(|x| *x = 1);

        // Create collision groups
        let mut collision_groups = CollisionGroups::new(usize::from(rings));

        // Create random points
        let mut points: Vec<Vec3> = Vec::new();
        points.push(Vec3::new(0.0, 0.0, 0.0));
        collision_groups.add_point(0, [0, 0, 0]);
        for (ring, ring_count) in ring_counts[1..].iter().copied().enumerate() {
            for i in 0..ring_count {
                let angle = (i as f64) / (ring_count as f64) * std::f64::consts::TAU;
                let distance = (ring as f64) + 1.0;
                let x = distance * angle.cos();
                let y = distance * angle.sin();
                let z = rng.f64() * 0.2 - 0.1;
                points.push(Vec3::new(x, y, z));
                collision_groups.add_point(points.len() - 1, [truncate(x), truncate(y), truncate(z)]);
            }
        }
        //println!("Groups: {:?}", collision_groups.groups);

        let mut pairs: Vec<[usize; 2]> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::new();

        let mut cusps: Vec<bool> = vec![false; points.len()];

        // Manually prepare first ring to help the generation algorithm
        for i in 1..=ARGS.triangles_per_point {
            pairs.push([i, i % ARGS.triangles_per_point + 1]); // ring
            pairs.push([0, i]); // spoke
            tris.push([0, i, i % ARGS.triangles_per_point + 1]);
        }

        // generate every ring from 2 to n
        println!("Creating rings");
        for ring in 2..=rings.into() {
            let offset: usize = ring_counts[..ring].iter().sum();
            let prev_offset: usize = ring_counts[..ring - 1].iter().sum();
            let mut cur_previous: usize = offset - 1;
            let mut to_next_cusp: usize = 0;
            for i in 0..ring_counts[ring] {
                let index = i + offset;
                if to_next_cusp == 0 {
                    cusps[index] = true;
                    pairs.push([index, cur_previous]); // spoke

                    let temp =
                        (cur_previous - prev_offset + 1) % ring_counts[ring - 1] + prev_offset;
                    pairs.push([index, temp]); // spoke
                    tris.push([index, cur_previous, temp]);
                    cur_previous = temp;
                    to_next_cusp =
                        ARGS.triangles_per_point - if cusps[cur_previous] { 6 } else { 5 };
                } else {
                    pairs.push([index, cur_previous]); // spoke
                    to_next_cusp -= 1;
                }
                let next_index = (index - offset + 1) % ring_counts[ring] + offset;
                pairs.push([index, next_index]); // ring
                tris.push([index, next_index, cur_previous]);
            }
        }

        // Pull duals from existing pairs
        // TODO make this not O(n^3)
        // pairs.iter_mut().for_each(|p| p.sort());
        // pairs.sort();

        Mesh {
            points,
            pairs,
            collision_groups,
            tris,
        }
    }

    /// Does one iteration. For every point, it they are further apart
    ///  than 1, we move them closer together. If they are closer together
    ///  we push them further apart. If after checking every distance,
    ///  we move no points, then we return `ControlFlow::Break(())`.
    pub fn do_iteration(&mut self) -> ControlFlow<(), ()> {
        // Align duals
        /*let duals_passed = self
            .duals
            .iter()
            .copied()
            .map(|[a, b]| {
                if let ControlFlow::Continue((p1, p2)) =
                    move_duals(&self.points[a], &self.points[b])
                {
                    self.points[a] = p1;
                    self.points[b] = p2;
                    false
                } else {
                    true
                }
            })
            //Bitwise-and to certainly prevent short-circuit behavior
            .fold(true, |acc, elem| acc & elem);
        */

        // Align within collision groups
        let coll_passed = (0..self.points.len())
            .map(|a| {
                let c = *self.points[a];
                let group = [truncate(c.x), truncate(c.y), truncate(c.z)];
                self.collision_groups
                    .get_neighbor_points(a, group)
                    .iter()
                    .copied()
                    //.filter(|b| !self.pairs.contains(&[a, *b]))
                    .map(|b| {
                        if let ControlFlow::Continue((p1, p2)) =
                            move_duals(&self.points[a], &self.points[b])
                        {
                            //println!("Pair ({a}, {b})");
                            self.collision_groups.move_point(a, self.points[a], p1);
                            self.collision_groups.move_point(b, self.points[b], p2);
                            self.points[a] = p1;
                            self.points[b] = p2;
                            false
                        } else {
                            true
                        }
                    })
                    .fold(true, |acc, elem| acc & elem)
            })
            .fold(true, |acc, elem| acc & elem);
        // Align points of triangles
        let points_passed = self
            .pairs
            .iter()
            .copied()
            .map(|[a, b]| {
                if let ControlFlow::Continue((p1, p2)) =
                    move_points(&self.points[a], &self.points[b])
                {
                    self.collision_groups.move_point(a, self.points[a], p1);
                    self.collision_groups.move_point(b, self.points[b], p2);
                    self.points[a] = p1;
                    self.points[b] = p2;
                    false
                } else {
                    true
                }
            })
            //Bitwise-and to certainly prevent short-circuit behavior
            .fold(true, |acc, elem| acc & elem);
        if points_passed && coll_passed {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    pub fn test_collisions(&self) -> bool {
        for tri1 in self.tris.iter().copied() {
            for tri2 in self.tris.iter().copied() {
                if tri2.iter().any(|x| tri1.contains(x)) {
                    // Don't need to test triangles that share an edge
                    continue;
                }
                if collision(self.get_tri(tri1), self.get_tri(tri2)) {
                    // return true on collision
                    return true;
                }
            }
        }
        false
    }

    /// Converts the mesh into a vector of points of the triangles
    pub fn get_tri(&self, indicies: [usize; 3]) -> [Vec3; 3] {
        [
            self.points[indicies[0]],
            self.points[indicies[1]],
            self.points[indicies[2]],
        ]
    }

    fn print_debug(&self) {
        for [a, b] in self.pairs.iter().copied() {
            println!(
                "Distance: {a}, {b} = {}",
                distance(&self.points[a], &self.points[b])
            );
        }

        for [a, b, c] in self.tris.iter().copied() {
            let dist1 = distance(&self.points[a], &self.points[b]);
            let dist2 = distance(&self.points[b], &self.points[c]);
            let dist3 = distance(&self.points[c], &self.points[a]);

            println!("Triangle: {a}, {b}, {c}");
            println!("Dists: {dist1}, {dist2}, {dist3}");
        }
    }
}

// fn sphere_rand(radius: f64) -> Vec3 {
//     let mut rng = thread_rng();

//     // Generate two random numbers between 0 and 1
//     let u = rng.gen::<f64>();
//     let v = rng.gen::<f64>();

//     // Calculate the longitude and latitude
//     let lon = 2.0 * std::f64::consts::PI * u;
//     let lat = f64::acos(2.0_f64.mul_add(v, -1.0)) - std::f64::consts::FRAC_PI_2;

//     // Calculate the x, y, and z coordinates of the point
//     let x = f64::cos(lon) * f64::cos(lat) * radius;
//     let y = f64::sin(lon) * f64::cos(lat) * radius;
//     let z = f64::sin(lat) * radius;
//     Vec3::new(x, y, z)
// }

fn move_points(p1: &Vec3, p2: &Vec3) -> ControlFlow<(), (Vec3, Vec3)> {
    // Caculate the distance between the two points
    let dist = distance(p1, p2);
    // If the distance is close enough to the error margin
    if (dist - 1.0).abs() < ARGS.error_margin {
        return ControlFlow::Break(());
    }
    // Else, we move them by 1/10th the distance with a touch of randomness
    let scalar = (1.0 - dist) * 0.1;
    let offset1 = (*p1 - *p2).scale(scalar); // + sphere_rand(scalar * 0.7);
    let offset2 = (*p2 - *p1).scale(scalar); // + sphere_rand(scalar * 0.7);
    ControlFlow::Continue((*p1 + offset1, *p2 + offset2))
}

fn move_duals(p1: &Vec3, p2: &Vec3) -> ControlFlow<(), (Vec3, Vec3)> {
    // Caculate the distance between the two points
    let min_dist = 1.0; // TODO Add as argument
    let dist = distance(p1, p2);
    if dist > min_dist {
        return ControlFlow::Break(());
    }
    let scalar = (min_dist - dist) * 0.6;
    let offset1 = (*p1 - *p2).scale(scalar); // + sphere_rand(scalar * 0.7);
    let offset2 = (*p2 - *p1).scale(scalar); // + sphere_rand(scalar * 0.7);
    ControlFlow::Continue((*p1 + offset1, *p2 + offset2))
}

fn ray_triangle(
    ray_origin: &Vec3,
    ray_vector: &Vec3,
    vertex0: &Vec3,
    vertex1: &Vec3,
    vertex2: &Vec3,
) -> bool {
    let edge1 = *vertex1 - *vertex0;
    let edge2 = *vertex2 - *vertex0;
    let h = ray_vector.coords.cross(&edge2);
    let a = edge1.dot(&h);
    if a > -f64::EPSILON && a < f64::EPSILON {
        // The ray is parallel
        return false;
    }
    let f = 1.0 / a;
    let s = *ray_origin - *vertex0;
    let u = f * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(&edge1);
    let v = f * ray_vector.coords.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = f * edge2.dot(&q);
    if t > f64::EPSILON
    // ray intersection
    {
        return true;
    }
    false
}

fn collision_partial(tri1: [Vec3; 3], tri2: [Vec3; 3]) -> bool {
    let vertex0 = tri1[0];
    let vertex1 = tri1[1];
    let vertex2 = tri1[2];

    let origin0 = tri2[0];
    let origin1 = tri2[1];
    let origin2 = tri2[2];
    (ray_triangle(
        &origin0,
        &(origin1 - origin0).into(),
        &vertex0,
        &vertex1,
        &vertex2,
    ) && ray_triangle(
        &origin1,
        &(origin0 - origin1).into(),
        &vertex0,
        &vertex1,
        &vertex2,
    )) || (ray_triangle(
        &origin0,
        &(origin2 - origin0).into(),
        &vertex0,
        &vertex1,
        &vertex2,
    ) && ray_triangle(
        &origin2,
        &(origin0 - origin2).into(),
        &vertex0,
        &vertex1,
        &vertex2,
    )) || (ray_triangle(
        &origin1,
        &(origin2 - origin1).into(),
        &vertex0,
        &vertex1,
        &vertex2,
    ) && ray_triangle(
        &origin2,
        &(origin1 - origin2).into(),
        &vertex0,
        &vertex1,
        &vertex2,
    ))
}

fn collision(tri1: [Vec3; 3], tri2: [Vec3; 3]) -> bool {
    collision_partial(tri1, tri2) || collision_partial(tri2, tri1)
}

fn main() {
    println!(
        "Creating mesh with {} rings and {} triangles per point",
        ARGS.rings, ARGS.triangles_per_point
    );

    println!("Creating points");
    if ARGS.animate {
        match std::fs::remove_dir_all("./output/") {
            Ok(()) => {}
            Err(x) if x.kind() == ErrorKind::NotFound => {}
            Err(x) => panic!("Err: {x}"),
        }
        std::fs::create_dir("./output").unwrap();
    }
    // Create the new mesh with the user given ring count
    let mut mesh = Mesh::new(ARGS.rings);

    let stop = Arc::new(AtomicBool::new(false));

    let stop_signal = stop.clone();
    //let signal_mesh = mesh.clone();
    let ctrlc_err = ctrlc::set_handler(move || {
        println!("Recieved interrupt. Exiting early");
        stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    if ctrlc_err.is_err() {
        println!("Failed to set ctrl-c hook");
    }

    println!("Moving points into proper place");
    // Start a timer to have time bounded ending
    let mut start = Instant::now();

    // Create a progress bar for the user
    let pb = ProgressBar::new(ARGS.time().as_millis() as u64);
    pb.set_style(
        ProgressStyle::with_template("{percent}%  {wide_bar}  [{elapsed_precise}]").unwrap(),
    );

    let mut frame_number = 0;
    let mut attempts = 0;
    // While the timer hasn't surpassed the user given time-limit
    while start.elapsed() < ARGS.time() && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        assert!(mesh.points.iter().all(|p| p.iter().all(|c| c.is_finite())));

        if start.elapsed() > Duration::from_secs(5) && mesh.test_collisions() {
            // Restart if colliding
            attempts += 1;
            pb.suspend(|| {
                println!("Restarting due to collision. Attempts: {attempts}");
                mesh = Mesh::new(ARGS.rings);
                frame_number = 0;
                start = Instant::now();
            });
            pb.reset_elapsed();
        }

        if ARGS.animate {
            save_mesh(&mesh, format!("./output/frame-{frame_number:07}.obj"));
            frame_number += 1;
        }
        // We move the points
        if let ControlFlow::Break(_) = mesh.do_iteration() {
            if ARGS.animate {
                save_mesh(&mesh, format!("./output/frame-{frame_number:07}.obj"));
            }
            // If we didn't move any, we are within error margins and we can exit
            pb.println("Stopping as we are within error margins");
            break;
        }
        /*if mesh.test_collisions() {
            pb.println("Stopping due to collision");
            break;
        }*/
        // Otherwise, update the progress bar, and continue
        pb.set_position(start.elapsed().as_millis() as u64);
    }
    pb.finish();

    // Debug print the mesh with every distance and point
    if ARGS.debug {
        mesh.print_debug();
    }

    if mesh.test_collisions() {
        println!("Mesh is self-intersecting");
    }

    // This was for outputting to csv, which we may come back to
    /*
    let mut output_file = File::create("./output.csv").unwrap();
    let tris = mesh.get_tris();
    for (a, b, c) in tris {
        let point1 = a.to_string();
        let point2 = b.to_string();
        let point3 = c.to_string();
        writeln!(output_file, "{point1:<40},{point2:<40},{point3:<40}").unwrap();
    }
    println!("Successfully created output.csv");
    */

    save_mesh(&mesh, "./output.obj".into());
}

fn save_mesh(mesh: &Mesh, filename: String) {
    // Convert the Vec<Point> into Vec<[f64; 3]>
    let verts = mesh
        .points
        .iter()
        .copied()
        .map(|p| [p.x, p.y, p.z])
        .collect::<Vec<_>>();

    // Create a mesh object
    let tri_mesh = TriMesh::new(verts, mesh.tris.clone());
    // And save that object into an .obj file
    match save_trimesh_ascii(&tri_mesh, filename) {
        Ok(()) => {}
        Err(x) => panic!("Error: {x}"),
    }
}
