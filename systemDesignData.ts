
import type { Question } from '@/types';

export const SYSTEM_DESIGN_QUESTIONS: Question[] = [
  {
    id: '1',
    title: 'Design a URL Shortener (TinyURL)',
    difficulty: 'Easy',
    category: 'System Design',
    tags: ['System Design', 'Database', 'Hashing', 'Scalability'],
    companies: ['Google', 'System Design'],
    description: `Design a service like TinyURL that converts long URLs into short, unique aliases.
    
Key Requirements:
1. Given a long URL, return a unique short URL.
2. When clicking the short URL, redirect to the original long URL.
3. Short links should not expire (or have custom expiration).
4. Highly available and low latency for redirection.

Considerations:
- Traffic estimation (Read vs Write ratio)
- Storage estimation
- Database selection
- Hashing algorithm
- Collision handling`,
    officialSolution: `
High-Level Architecture:

1. API Endpoints:
   - POST /api/shorten (input: longUrl) -> returns shortUrl
   - GET /:shortUrl -> 301 Redirect to longUrl

2. Capacity Planning:
   - Writes are low compared to Reads (e.g., 1:100 ratio).
   - Need huge storage for billions of records.

3. Database Schema:
   - Table: URLs (id, long_url, short_url, created_at, expires_at)
   - DB Choice: NoSQL (Cassandra/DynamoDB) for high scalability and write speeds, or Relational (Postgres) if strictly consistent.

4. Shortening Logic:
   - Base62 Encoding (A-Z, a-z, 0-9) = 62 chars.
   - A 7-character key allows 62^7 (~3.5 trillion) combinations.
   - Strategy: Pre-generate keys (Key Generation Service - KGS) or Hash(LongURL) + Base62.

5. Handling Concurrency:
   - If using a counter, use a distributed unique ID generator (Twitter Snowflake) or a dedicated KGS that loads unused keys into memory.

6. Caching:
   - Redis/Memcached to cache "shortUrl -> longUrl" mappings.
   - Eviction Policy: LRU (Least Recently Used).
    `
  },
  {
    id: '2',
    title: 'Design a Rate Limiter',
    difficulty: 'Medium',
    category: 'System Design',
    tags: ['System Design', 'API', 'Algorithms', 'Security'],
    companies: ['Stripe', 'Google', 'Meta'],
    description: `Design an API Rate Limiter that throttles users based on the number of requests they send within a specified time window.

Key Requirements:
1. Accurately limit requests (e.g., 10 requests per second).
2. Low latency (the limiter itself shouldn't slow down the API).
3. Distributed environment support (multiple servers).

Considerations:
- Where to place the limiter (Client vs Server vs Middleware).
- Algorithms (Token Bucket, Leaking Bucket, Fixed Window, Sliding Window Log).`,
    officialSolution: `
Approaches:

1. Token Bucket Algorithm:
   - A bucket holds 'n' tokens. Tokens replenish at a fixed rate.
   - Each request consumes a token. If empty, request is dropped.
   - Pros: Allows bursts of traffic, memory efficient.

2. Leaking Bucket:
   - Requests enter a queue processed at a constant rate.
   - Pros: Smooths out traffic. Cons: Bursts can fill queue and drop requests.

3. Fixed Window Counter:
   - Count requests in 1-minute windows.
   - Issue: A burst at the end of min 1 and start of min 2 can double the allowed rate (boundary problem).

4. Sliding Window Counter (Hybrid):
   - Weighted calculation of previous window + current window.
   - Most practical for strict rate limiting.

Architecture:
- Store counters in Redis (fast, supports atomic increment/expiry).
- Middleware intercepts request -> checks Redis -> allows or returns 429 Too Many Requests.
    `
  },
  {
    id: '3',
    title: 'Design a Web Crawler',
    difficulty: 'Hard',
    category: 'System Design',
    tags: ['System Design', 'Distributed Systems', 'Networking'],
    companies: ['Google', 'Microsoft'],
    description: `Design a scalable web crawler (like Googlebot) that collects information from the entire web.

Key Requirements:
1. Scalability: Must handle billions of pages.
2. Politeness: Do not overwhelm target servers.
3. Extensibility: Support new content types (images, videos).
4. Robustness: Handle malformed HTML, server crashes, etc.

Considerations:
- URL Frontier (queue management).
- DNS Resolution.
- Deduplication.`,
    officialSolution: `
Core Components:

1. Seed URLs: Starting point.

2. URL Frontier:
   - Prioritized queue of URLs to visit.
   - Ensures politeness (delay between requests to same host).
   - Split into multiple queues based on hostname.

3. HTML Downloader & DNS Resolver:
   - Fetches page content. Needs high-performance DNS caching (since standard DNS is slow).

4. Content Parser:
   - Validates and parses HTML. Extracts new links.

5. Deduplication (Content & URL):
   - Fingerprint (Checksum/MD5) stored in DB to check if page content was already seen.
   - Bloom Filter for fast "Has this URL been seen?" check.

6. Storage:
   - BigTable or HBase for storing simplified page content.

Workflow:
Frontier -> Downloader -> Parser -> Dedup -> Storage -> (Extracted Links) -> Frontier.
    `
  },
  {
    id: '4',
    title: 'Design Instagram (News Feed)',
    difficulty: 'Hard',
    category: 'System Design',
    tags: ['System Design', 'Social Network', 'Feed Architecture', 'Database'],
    companies: ['Meta', 'Google'],
    description: `Design a photo-sharing application like Instagram where users can upload photos and follow other users to see a generated news feed.

Key Requirements:
1. Users can upload photos.
2. Users can follow others.
3. Generate a timeline (feed) of photos from followed users.
4. Low latency for generating the feed.

Considerations:
- Read vs Write Heavy? (Read heavy).
- Pull vs Push model for feeds.`,
    officialSolution: `
Architecture:

1. Database Schema:
   - Users, Photos (metadata), Follows (user_id, follower_id).
   - Photo Storage: S3 or object storage (CDN for delivery).
   - Metadata Storage: SQL (MySQL/Postgres) sharded by UserID.

2. Feed Generation Models:

   A. Pull Model (Fan-out-on-load):
   - User opens app -> Query DB for all followees -> Fetch latest photos -> Merge & Sort.
   - Pros: Simple write.
   - Cons: Slow read for users following many people.

   B. Push Model (Fan-out-on-write):
   - User uploads photo -> System pushes photo ID to all followers' pre-computed feed lists (in Redis).
   - User opens app -> Simply reads from their Redis list.
   - Pros: Lightning fast reads.
   - Cons: "Celebrity problem" (writing to millions of followers is slow).

3. Hybrid Solution:
   - Push for normal users.
   - Pull for celebrities (don't push their content; fetch it when followers load feed).

4. Reliability:
   - Master-Slave replication for databases.
    `
  },
  {
    id: '5',
    title: 'Design a Key-Value Store (like DynamoDB)',
    difficulty: 'Hard',
    category: 'System Design',
    tags: ['System Design', 'Distributed Databases', 'Consistency', 'CAP Theorem'],
    companies: ['Amazon', 'Google', 'Microsoft'],
    description: `Design a distributed Key-Value store that is highly available, scalable, and partition-tolerant.

Key Requirements:
1. put(key, value)
2. get(key)
3. Configurable consistency (Strong vs Eventual).
4. Handle node failures gracefully.

Considerations:
- CAP Theorem.
- Data Partitioning (Consistent Hashing).
- Replication.`,
    officialSolution: `
Core Concepts:

1. Data Partitioning:
   - Consistent Hashing (Ring).
   - Distributes data evenly across nodes.
   - Virtual nodes to handle hotspots/heterogeneous hardware.

2. Data Replication:
   - Replicate data to N nodes (e.g., N=3) moving clockwise on the ring.
   - Preference List: The list of nodes responsible for a key.

3. Consistency (Quorum):
   - R + W > N.
   - W = writes must be acknowledged by W nodes.
   - R = reads must be confirmed by R nodes.
   - Tunable: W=1 (Fast, risky), W=All (Slow, strong consistency).

4. Conflict Resolution:
   - Vector Clocks to detect version conflicts.
   - Last-Write-Wins (LWW) for simplicity (used by Cassandra).

5. Failure Detection:
   - Gossip Protocol: Nodes periodically exchange state info to detect down nodes quickly.
   - Merkle Trees: To detect data inconsistencies between replicas during anti-entropy (repair).
    `
  },
  {
    id: '6',
    title: 'Design WhatsApp (Chat App)',
    difficulty: 'Medium',
    category: 'System Design',
    tags: ['System Design', 'Real-time', 'WebSockets', 'Database'],
    companies: ['Meta', 'Google'],
    description: `Design a real-time chat application like WhatsApp or Facebook Messenger.

Key Requirements:
1. 1-on-1 Chat.
2. Group Chat (max 100 people).
3. Sent, Delivered, and Read receipts.
4. Online/Offline Status.
5. Messages should be persistent.

Considerations:
- Protocol (HTTP vs WebSocket).
- Database for chat history (Read/Write pattern).
- Push Notifications for offline users.`,
    officialSolution: `
Architecture:

1. Connection Handling:
   - Use WebSockets for real-time bi-directional communication.
   - Chat Server maintains open connections with online users.

2. Message Flow (A sends to B):
   - A -> Load Balancer -> Chat Server -> B (if online).
   - If B is offline: Store in DB -> Push Notification Service -> APNS/FCM.

3. Database Choice:
   - HBase or Cassandra (Wide-column stores).
   - Pattern: High write volume. Queries are usually "Get last 50 messages for conversation ID".
   - RDBMS is hard to scale for billions of messages/day.

4. Group Chat:
   - Client sends msg to Group Service.
   - Service looks up group members.
   - Fan-out message to all members' connected Chat Servers.

5. Asset Management:
   - Images/Videos uploaded to Blob Storage (S3).
   - URL sent in the chat message.
`
  },
  {
    id: '7',
    title: 'Design YouTube (Video Streaming)',
    difficulty: 'Hard',
    category: 'System Design',
    tags: ['System Design', 'Streaming', 'CDN', 'Storage'],
    companies: ['Google', 'Netflix', 'Twitch'],
    description: `Design a video sharing and streaming platform like YouTube or Netflix.

Key Requirements:
1. Upload videos.
2. Watch videos (smooth streaming, adaptive quality).
3. View statistics (view count).
4. Search videos.

Considerations:
- Video storage and bandwidth.
- Transcoding (converting video to different formats/resolutions).
- Content Delivery Network (CDN).`,
    officialSolution: `
Architecture:

1. Upload Flow:
   - User -> Load Balancer -> Web Server -> Original Storage (S3).
   - Trigger message to Message Queue (Kafka/RabbitMQ) for processing.

2. Video Processing (Transcoding):
   - Workers pull jobs from Queue.
   - Convert video into multiple formats (MP4, HLS) and resolutions (360p, 720p, 1080p).
   - Store processed chunks in S3.

3. Streaming (Read Flow):
   - Use CDN (Cloudfront/Akamai) to cache video chunks geographically closer to users.
   - Adaptive Bitrate Streaming (HLS/DASH): Client switches quality based on bandwidth.

4. Metadata & Search:
   - Video metadata (title, description) in SQL (Sharded) or NoSQL.
   - Search index built using ElasticSearch (updated via stream from DB).

5. Deduplication:
   - Hash entire video file to check duplicates before processing.
`
  },
  {
    id: '8',
    title: 'Design Uber (Ride Sharing)',
    difficulty: 'Hard',
    category: 'System Design',
    tags: ['System Design', 'Geo-Spatial', 'Real-time', 'Matching'],
    companies: ['Uber', 'Grab', 'Lyft'],
    description: `Design a ride-hailing service like Uber or Lyft.

Key Requirements:
1. Riders can request a ride.
2. Drivers get notified of nearby requests.
3. Real-time location tracking of driver.
4. ETA calculation.

Considerations:
- How to store and query location data efficiently.
- Matching algorithm.
- Handling high concurrency of location updates.`,
    officialSolution: `
Architecture:

1. Location Storage (Geo-Spatial):
   - Drivers send updates every 4 seconds.
   - Database: Redis (Geo commands) for active drivers (highly ephemeral data).
   - Persistent storage: Cassandra (for trip history).

2. Indexing Locations:
   - QuadTree or Google S2 Geometry.
   - Divides map into grids. Fast lookup for "Find drivers in Grid X".

3. Matching Service:
   - Rider requests ride.
   - Service queries Redis/QuadTree for nearby drivers.
   - Locks driver to prevent double booking.
   - Sends notification to Driver.

4. Communication:
   - WebSockets for real-time location updates on Rider's map.
   - Push Notifications for "Driver Arrived".

5. Trip Management:
   - State Machine: Requested -> Matched -> Picked Up -> Dropped Off.
   - Handled by Trip Service backed by SQL database (transactional integrity required).
`
  },
  {
    id: '9',
    title: 'Design Typeahead (Search Autocomplete)',
    difficulty: 'Medium',
    category: 'System Design',
    tags: ['System Design', 'Search', 'Data Structures', 'Trie'],
    companies: ['Google', 'Amazon', 'Microsoft'],
    description: `Design a search autocomplete system (like Google Search bar).

Key Requirements:
1. Fast response time (< 100ms).
2. Results should be relevant (sorted by popularity).
3. Handle huge scale of queries.

Considerations:
- Data Structure for prefix search.
- Where to store the index.
- Updating the index with new trending searches.`,
    officialSolution: `
Architecture:

1. Data Structure:
   - Trie (Prefix Tree).
   - Each node stores the character + top 5 most searched terms ending at that node (cache top results).

2. Storage:
   - In-Memory (Redis) for extremely fast lookups.
   - Persistent DB (Cassandra/DynamoDB) to rebuild Trie on restart.

3. Write Path (Data Gathering):
   - Analytics logs search queries.
   - Aggregation Service counts frequencies (e.g., MapReduce/Spark streaming).
   - Updates the Trie periodically (e.g., every hour) or in near real-time.

4. Read Path:
   - User types "sys".
   - LB -> Autocomplete Service.
   - Service traverses Trie for "sys" -> returns cached top results ["system design", "system failure", ...].

5. Optimization:
   - Browser caching (Cache-Control headers) for short prefixes.
   - Sampling: Only log 1% of search queries to save resources.
`
  },
  {
    id: '10',
    title: 'Design a Notification System',
    difficulty: 'Medium',
    category: 'System Design',
    tags: ['System Design', 'Messaging', 'Queue'],
    companies: ['Amazon', 'Apple', 'Netflix'],
    description: `Design a centralized notification system that sends emails, SMS, and Push Notifications to users.

Key Requirements:
1. Support pluggable providers (SendGrid, Twilio, FCM/APNS).
2. Rate limiting (don't spam users).
3. Prioritization (OTP > Marketing).
4. Retry mechanism for failed notifications.

Considerations:
- Interface definition.
- Queue isolation.
- Worker architecture.`,
    officialSolution: `
Architecture:

1. API Gateway:
   - POST /send (userId, type, content, priority).

2. Notification Service:
   - Validates request.
   - Checks User Preferences (Opt-in/out).
   - Checks Rate Limits (Redis counter).

3. Message Queues (Kafka/RabbitMQ):
   - Separate queues for priorities: 'Critical', 'High', 'Low'.
   - Separate topics for channels: 'SMS', 'Email', 'Push'.

4. Workers:
   - Pull messages from queues.
   - Call third-party APIs (Twilio, SendGrid).
   - If API fails, push to 'Retry Queue' with exponential backoff.

5. Observability:
   - Logs status (Sent, Failed, Delivered) to a database for tracking and debugging.
`
  }
];
