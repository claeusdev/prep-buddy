import React, { useState } from 'react';
import type { ReferenceItem } from '@/types';
import ReferenceCard from './ReferenceCard';
import { Search } from 'lucide-react';

interface ReferenceLibraryProps {
  items: ReferenceItem[];
  title: string;
}

const ReferenceLibrary: React.FC<ReferenceLibraryProps> = ({ items, title }) => {
  const [flippedId, setFlippedId] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  const filteredItems = items.filter(item => 
    item.title.toLowerCase().includes(search.toLowerCase()) ||
    item.summary.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-8 max-w-7xl mx-auto w-full">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
        <h1 className="text-3xl font-black font-mono uppercase tracking-tight border-l-4 border-black pl-4">{title}</h1>
        <div className="relative w-full md:w-80">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-black" />
          <input 
            type="text" 
            placeholder={`SEARCH_INDEX...`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white text-black border-2 border-black font-mono text-sm focus:outline-none focus:bg-yellow-50 shadow-retro-sm"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {filteredItems.map(item => (
          <div key={item.id} onClick={() => setFlippedId(flippedId === item.id ? null : item.id)}>
            <ReferenceCard 
              item={item}
              isFlipped={flippedId === item.id}
              onFlip={() => setFlippedId(flippedId === item.id ? null : item.id)}
            />
          </div>
        ))}
      </div>
      
      {filteredItems.length === 0 && (
        <div className="text-center py-20 text-gray-400 font-mono">
          <p>ERR: NO_MATCHING_ITEMS "{search}"</p>
        </div>
      )}
    </div>
  );
};

export default ReferenceLibrary;